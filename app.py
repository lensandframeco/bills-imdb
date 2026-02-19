import streamlit as st
import anthropic
import base64
import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

st.set_page_config(page_title="Bill's IMDB", layout="wide")
st.title("Bill's IMDB")

SYSTEM_PROMPT = """\
You are an expert TV production research assistant helping entertainment \
industry professionals. You have deep knowledge of television history, \
showrunners, writers rooms, network executives, production companies, and \
the business side of TV.

RESEARCH APPROACH:
You have access to multiple research tools. Be efficient and targeted.

CRITICAL RULES:
- Use at most 4-5 tool calls total per question, then synthesize your answer.
- Call multiple tools in parallel when possible (counts as one round).
- ALWAYS present what you found. NEVER ask the user to "help narrow it down" \
or give "possible explanations" for missing data. Present your results \
confidently and completely, even if they come only from web search.
- NEVER suggest the user go check IMDbPro, Deadline, Variety, or any other \
source themselves. YOU have search tools for all of these -- use them. \
Do not tell the user to look something up that you could search for. \
If your searches didn't find something, just say it wasn't found -- do \
NOT redirect the user to go search on their own.
- NEVER say someone has "limited credits" or is "early in their career" \
just because TMDB has sparse data. TMDB is NOT comprehensive. Many working \
professionals (agents, managers, below-the-line crew, composers, music \
supervisors) have extensive IMDB credits that TMDB doesn't track.

Tool strategy:
1. **First round (parallel):** Call ALL of these at once:
   - tmdb_search to find their TMDB ID
   - web_search: "[name] site:imdb.com" to find their IMDB page
   - trades_search: "[name] new project 2025 2026" to find upcoming \
work, deals, castings, and buzz in the trades
2. **Second round:** Use tmdb_tv_details / tmdb_person_credits if TMDB \
returned good results. If you need more on upcoming work, call \
trades_search with "[name] cast upcoming series pilot" or similar.
3. Combine ALL sources into one comprehensive answer. Trade publication \
articles and web search results are just as valid as TMDB data.

UPCOMING PROJECTS & CURRENT BUZZ:
For EVERY person query, always research what they're doing NOW and NEXT:
- Search trades for recent castings, signings, deals, and announcements.
- Include an "Upcoming / In Development" section in your answer when \
relevant info is found.
- Mention recent press buzz, festival appearances, award nominations, \
or social media attention if found in search results.
- For actors: look for new series, pilots, or films they've been cast in.
- For writers/showrunners: look for overall deals, new series orders, \
pilots in development.
- For executives: look for recent promotions, moves, greenlight decisions.
- If the person just came off a hit show or breakout role, emphasize \
that context and what's next for them.

MISSPELLINGS AND FUZZY MATCHING:
Users will often misspell names. This is expected.
- Use tmdb_search with the misspelled name -- TMDB handles fuzzy matching.
- Identify the most likely correct match from the results.
- State your best match: "I believe you're referring to **[correct name]**."
- Then proceed to answer fully. NEVER refuse due to a spelling issue.

RESUME ANALYSIS:
When the user uploads a resume or CV, analyze it thoroughly:
1. Identify the person and extract every TV show, production company, \
network, and role mentioned in the resume.
2. For each show or company, use tmdb_search and tmdb_tv_details to find \
who else worked on those projects (other writers, showrunners, EPs, cast).
3. Organize your answer chronologically by career stage: early career, \
mid-career, and current/recent. For each stage, list key collaborators.
4. Highlight repeat collaborators -- people who appear across multiple \
projects with this person.
5. If the resume mentions a person by name, research them too.
6. If the resume is NOT from the entertainment industry (no TV shows, \
films, production companies, networks, or entertainment roles), tell the \
user politely: "This doesn't appear to be a film/TV production resume. \
This tool is designed for entertainment industry research. I can see \
this person works in [their actual field]." Then offer to answer any \
general questions they have about the document.

RESPONSE STYLE:
- Lead with the answer. Most important facts first.
- Use structured formatting: headers, bullet points, bold for key names.
- Include specific dates, seasons, and episode counts.
- For executive/network questions, include title and tenure dates.
- Distinguish between created by, showrunner, writer, EP, and other roles.
- Always include a **What's Next** or **Upcoming** section when there is \
any info about future projects, deals, or announced roles.
- Cite sources inline when pulling facts from search results.
- If uncertain about a detail, say so briefly and give what you do know.\
"""

TMDB_BASE = "https://api.themoviedb.org/3"

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for information about TV shows, entertainment "
        "industry people, production companies, network executives, ratings, "
        "awards, and industry news. Covers IMDb, Wikipedia, Deadline, "
        "Variety, THR, The Futon Critic, and more. Use for anything TMDB "
        "doesn't cover, like executive roles, deals, or industry analysis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query. Be specific: include names, show titles, "
                    "networks, years."
                ),
            }
        },
        "required": ["query"],
    },
}

TMDB_SEARCH_TOOL = {
    "name": "tmdb_search",
    "description": (
        "Search TMDB for TV shows or people by name. Returns matching "
        "results with TMDB IDs needed for detailed lookups. Handles "
        "misspellings well. Always start here before using other TMDB tools."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Name of the TV show or person.",
            },
            "type": {
                "type": "string",
                "enum": ["tv", "person"],
                "description": "Search for TV shows or people.",
            },
        },
        "required": ["query", "type"],
    },
}

TMDB_TV_DETAILS_TOOL = {
    "name": "tmdb_tv_details",
    "description": (
        "Get full details about a TV show from TMDB: overview, status, "
        "air dates, seasons, ratings, networks, production companies, "
        "creators, cast, and crew. Requires TMDB show ID from tmdb_search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "show_id": {
                "type": "integer",
                "description": "TMDB show ID from tmdb_search results.",
            }
        },
        "required": ["show_id"],
    },
}

TMDB_PERSON_CREDITS_TOOL = {
    "name": "tmdb_person_credits",
    "description": (
        "Get a person's full TV career from TMDB: biography, every TV show "
        "they worked on as cast or crew, with role details and episode "
        "counts. Requires TMDB person ID from tmdb_search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "person_id": {
                "type": "integer",
                "description": "TMDB person ID from tmdb_search results.",
            }
        },
        "required": ["person_id"],
    },
}

TMDB_SEASON_DETAILS_TOOL = {
    "name": "tmdb_season_details",
    "description": (
        "Get episode-level details for a specific season: episode names, "
        "air dates, ratings, per-episode writer and director credits. "
        "Requires TMDB show ID and season number."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "show_id": {
                "type": "integer",
                "description": "TMDB show ID.",
            },
            "season_number": {
                "type": "integer",
                "description": "Season number (0 for specials).",
            },
        },
        "required": ["show_id", "season_number"],
    },
}

TRADES_SEARCH_TOOL = {
    "name": "trades_search",
    "description": (
        "Search entertainment trade publications: Deadline, Variety, "
        "Hollywood Reporter (THR), The Wrap, IndieWire, Backstage, and "
        "Broadcasting & Cable. Use for deals, hirings, firings, executive "
        "moves, agency news, pilot pickups, renewals, cancellations, and "
        "industry analysis. More targeted than general web search for "
        "entertainment industry news."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query. Include names, titles, companies. "
                    "E.g., 'John Smith showrunner deal' or "
                    "'Paradigm talent agency hires'."
                ),
            }
        },
        "required": ["query"],
    },
}

TRADE_SITES = [
    "deadline.com",
    "variety.com",
    "hollywoodreporter.com",
    "thewrap.com",
    "indiewire.com",
    "backstage.com",
    "broadcastingcable.com",
    "tvline.com",
]

MAX_TOOL_ROUNDS = 6


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
def execute_trades_search(query):
    """Search entertainment trade publications specifically."""
    site_filter = " OR ".join(f"site:{s}" for s in TRADE_SITES)
    full_query = f"{query} ({site_filter})"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(full_query, max_results=8))
        if not results:
            # Fallback: try without site filter but add "deadline variety THR"
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    f"{query} deadline variety hollywood reporter",
                    max_results=6,
                ))
        if not results:
            return "No trade publication results found. Try a different query."
        formatted = []
        for r in results:
            formatted.append(
                f"Source: {r['href']}\n"
                f"Title: {r['title']}\n"
                f"Snippet: {r['body']}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Trades search error: {e}. Try a different query."


def execute_web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
        if not results:
            return "No results found. Try a different query."
        formatted = []
        for r in results:
            formatted.append(
                f"Title: {r['title']}\n"
                f"Snippet: {r['body']}\n"
                f"URL: {r['href']}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {e}. Try a different query."


def tmdb_get(endpoint, token, params=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    resp = requests.get(
        f"{TMDB_BASE}{endpoint}", headers=headers, params=params or {},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def execute_tmdb_search(query, search_type, token):
    data = tmdb_get(f"/search/{search_type}", token, {"query": query})
    results = data.get("results", [])[:8]
    if not results:
        return f"No TMDB results for '{query}'. Try a different spelling."
    if search_type == "tv":
        lines = []
        for r in results:
            lines.append(
                f"ID: {r['id']} | {r['name']} "
                f"({r.get('first_air_date', 'N/A')}) | "
                f"Rating: {r.get('vote_average', 'N/A')}/10 | "
                f"{r.get('overview', '')[:150]}"
            )
        return "\n\n".join(lines)
    else:
        lines = []
        for r in results:
            known_for = ", ".join(
                k.get("name", k.get("title", ""))
                for k in r.get("known_for", [])
            )
            lines.append(
                f"ID: {r['id']} | {r['name']} | "
                f"Dept: {r.get('known_for_department', 'N/A')} | "
                f"Known for: {known_for}"
            )
        return "\n\n".join(lines)


def execute_tmdb_tv_details(show_id, token):
    show = tmdb_get(f"/tv/{show_id}", token)
    credits = tmdb_get(f"/tv/{show_id}/aggregate_credits", token)
    seasons_info = "\n".join(
        f"  S{s['season_number']}: {s.get('episode_count', '?')} eps "
        f"({s.get('air_date', 'N/A')})"
        for s in show.get("seasons", [])
    )
    networks = ", ".join(n["name"] for n in show.get("networks", []))
    companies = ", ".join(
        c["name"] for c in show.get("production_companies", [])
    )
    creators = ", ".join(c["name"] for c in show.get("created_by", []))
    genres = ", ".join(g["name"] for g in show.get("genres", []))
    cast = credits.get("cast", [])[:20]
    cast_info = "\n".join(
        f"  {c['name']} as "
        f"{c.get('roles', [{}])[0].get('character', 'N/A')} "
        f"({c.get('total_episode_count', '?')} eps)"
        for c in cast
    )
    crew = credits.get("crew", [])[:25]
    crew_info = "\n".join(
        f"  {c['name']} - "
        f"{c.get('jobs', [{}])[0].get('job', c.get('department', 'N/A'))} "
        f"({c.get('total_episode_count', '?')} eps)"
        for c in crew
    )
    return (
        f"SHOW: {show.get('name', 'N/A')}\n"
        f"Status: {show.get('status', 'N/A')}\n"
        f"First aired: {show.get('first_air_date', 'N/A')}\n"
        f"Last aired: {show.get('last_air_date', 'N/A')}\n"
        f"Seasons: {show.get('number_of_seasons', 'N/A')} | "
        f"Episodes: {show.get('number_of_episodes', 'N/A')}\n"
        f"Rating: {show.get('vote_average', 'N/A')}/10 "
        f"({show.get('vote_count', 0)} votes)\n"
        f"Networks: {networks}\n"
        f"Production: {companies}\n"
        f"Created by: {creators}\n"
        f"Genres: {genres}\n"
        f"Overview: {show.get('overview', 'N/A')}\n\n"
        f"SEASONS:\n{seasons_info}\n\n"
        f"TOP CAST:\n{cast_info}\n\n"
        f"KEY CREW:\n{crew_info}"
    )


def execute_tmdb_person_credits(person_id, token):
    person = tmdb_get(f"/person/{person_id}", token)
    credits = tmdb_get(f"/person/{person_id}/tv_credits", token)
    cast_credits = credits.get("cast", [])[:15]
    crew_credits = credits.get("crew", [])[:20]
    cast_info = "\n".join(
        f"  {c.get('name', 'N/A')} ({c.get('first_air_date', 'N/A')}) "
        f"as {c.get('character', 'N/A')} "
        f"({c.get('episode_count', '?')} eps)"
        for c in cast_credits
    )
    crew_info = "\n".join(
        f"  {c.get('name', 'N/A')} ({c.get('first_air_date', 'N/A')}) - "
        f"{c.get('job', c.get('department', 'N/A'))} "
        f"({c.get('episode_count', '?')} eps)"
        for c in crew_credits
    )
    bio = person.get("biography", "N/A")
    if len(bio) > 800:
        bio = bio[:800] + "..."
    return (
        f"PERSON: {person.get('name', 'N/A')}\n"
        f"Known for: {person.get('known_for_department', 'N/A')}\n"
        f"Birthday: {person.get('birthday', 'N/A')}\n"
        f"Bio: {bio}\n\n"
        f"TV ACTING CREDITS:\n{cast_info or '  None found'}\n\n"
        f"TV CREW CREDITS:\n{crew_info or '  None found'}"
    )


def execute_tmdb_season_details(show_id, season_number, token):
    season = tmdb_get(f"/tv/{show_id}/season/{season_number}", token)
    episodes = season.get("episodes", [])
    ep_lines = []
    for e in episodes:
        directors = ", ".join(
            d["name"] for d in e.get("crew", [])
            if d.get("job") == "Director"
        ) or "N/A"
        writers = ", ".join(
            w["name"] for w in e.get("crew", [])
            if w.get("job") == "Writer"
        ) or "N/A"
        ep_lines.append(
            f"  E{e['episode_number']:02d}: {e.get('name', 'N/A')} "
            f"({e.get('air_date', 'N/A')}) | "
            f"Rating: {e.get('vote_average', 'N/A')}/10 | "
            f"Dir: {directors} | Wri: {writers}"
        )
    overview = season.get("overview", "") or "N/A"
    if len(overview) > 300:
        overview = overview[:300] + "..."
    return (
        f"SEASON {season_number}: {season.get('name', 'N/A')}\n"
        f"Air date: {season.get('air_date', 'N/A')}\n"
        f"Episodes: {len(episodes)}\n"
        f"Overview: {overview}\n\n"
        f"EPISODES:\n" + "\n".join(ep_lines)
    )


def execute_tool(name, tool_input, tmdb_tok):
    try:
        if name == "web_search":
            return execute_web_search(tool_input.get("query", ""))
        elif name == "trades_search":
            return execute_trades_search(tool_input.get("query", ""))
        elif name == "tmdb_search":
            return execute_tmdb_search(
                tool_input["query"], tool_input["type"], tmdb_tok
            )
        elif name == "tmdb_tv_details":
            return execute_tmdb_tv_details(tool_input["show_id"], tmdb_tok)
        elif name == "tmdb_person_credits":
            return execute_tmdb_person_credits(
                tool_input["person_id"], tmdb_tok
            )
        elif name == "tmdb_season_details":
            return execute_tmdb_season_details(
                tool_input["show_id"], tool_input["season_number"], tmdb_tok
            )
        else:
            return f"Unknown tool: {name}"
    except requests.HTTPError as e:
        return f"TMDB API error: {e}. Check the ID and try again."
    except Exception as e:
        return f"Tool error: {e}"


def execute_tools_parallel(tool_blocks, tmdb_tok):
    """Execute multiple tool calls concurrently."""
    results = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {}
        for tb in tool_blocks:
            inp = tb.input if isinstance(tb.input, dict) else {}
            future = executor.submit(execute_tool, tb.name, inp, tmdb_tok)
            futures[future] = tb.id
        for future in as_completed(futures):
            tool_use_id = futures[future]
            try:
                results[tool_use_id] = future.result()
            except Exception as e:
                results[tool_use_id] = f"Error: {e}"
    return results


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    # Read API keys from env/secrets silently; only show input if not configured
    def _get_secret(key):
        val = os.environ.get(key)
        if val:
            return val
        try:
            return st.secrets[key]
        except (KeyError, FileNotFoundError):
            return ""

    _env_api_key = _get_secret("ANTHROPIC_API_KEY")
    if _env_api_key:
        api_key = _env_api_key
        st.caption("Anthropic API Key: configured")
    else:
        api_key = st.text_input("Anthropic API Key", type="password")

    model = st.selectbox(
        "Model",
        ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5"],
        index=0,
    )

    st.divider()
    st.header("TMDB API")
    _env_tmdb = _get_secret("TMDB_READ_TOKEN")
    if _env_tmdb:
        tmdb_token = _env_tmdb
        st.caption("Read Access Token: configured")
    else:
        tmdb_token = st.text_input(
            "Read Access Token",
            type="password",
            help="Free from themoviedb.org (Settings > API)",
        )

    st.divider()
    st.header("MCP Servers (Optional)")
    st.caption("Connect additional remote MCP servers.")

    with st.expander("Additional MCP Server"):
        mcp_url = st.text_input(
            "Server URL",
            key="mcp_url",
            value=os.environ.get("MCP_SERVER_URL", ""),
            placeholder="https://your-mcp-server.com/sse",
        )
        mcp_token_val = st.text_input(
            "Auth Token",
            key="mcp_token",
            type="password",
            value=os.environ.get("MCP_SERVER_TOKEN", ""),
        )

    active = ["Web Search", "Trades (THR/Deadline/Variety)"]
    if tmdb_token:
        active.append("TMDB")
    if mcp_url:
        active.append("MCP")
    st.success(f"Active: {', '.join(active)}")

    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_tools_and_mcp():
    tools = [WEB_SEARCH_TOOL, TRADES_SEARCH_TOOL]
    if tmdb_token:
        tools.extend([
            TMDB_SEARCH_TOOL, TMDB_TV_DETAILS_TOOL,
            TMDB_PERSON_CREDITS_TOOL, TMDB_SEASON_DETAILS_TOOL,
        ])
    mcp_servers = []
    mcp_tools = []
    if mcp_url:
        server = {"type": "url", "url": mcp_url, "name": "mcp-server"}
        if mcp_token_val:
            server["authorization_token"] = mcp_token_val
        mcp_servers.append(server)
        mcp_tools.append(
            {"type": "mcp_toolset", "mcp_server_name": "mcp-server"}
        )
    return tools, mcp_servers, mcp_tools


def extract_text(content_blocks):
    parts = []
    for block in content_blocks:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n\n".join(parts)


def tool_call_label(block):
    """Human-readable label for a tool call."""
    inp = block.input if isinstance(block.input, dict) else {}
    name = block.name
    if name == "web_search":
        return f"Web search: {inp.get('query', '')}"
    elif name == "trades_search":
        return f"Trades search: {inp.get('query', '')}"
    elif name == "tmdb_search":
        return f"TMDB search ({inp.get('type', '')}): {inp.get('query', '')}"
    elif name == "tmdb_tv_details":
        return f"TMDB show details (ID: {inp.get('show_id', '')})"
    elif name == "tmdb_person_credits":
        return f"TMDB person credits (ID: {inp.get('person_id', '')})"
    elif name == "tmdb_season_details":
        return (
            f"TMDB season {inp.get('season_number', '?')} "
            f"(show ID: {inp.get('show_id', '')})"
        )
    return f"Tool: {name}"


def display_tool_calls(content_blocks):
    """Show tool call expanders for non-text blocks."""
    for block in content_blocks:
        block_type = getattr(block, "type", None)

        if block_type == "thinking":
            with st.expander("Thinking"):
                st.markdown(block.thinking)

        elif block_type == "tool_use":
            with st.expander(tool_call_label(block)):
                st.json(
                    block.input if isinstance(block.input, dict) else {}
                )

        elif block_type == "mcp_tool_use":
            server = getattr(block, "server_name", "unknown")
            with st.expander(f"MCP: {block.name} (via {server})"):
                st.json(block.input)

        elif block_type == "mcp_tool_result":
            is_error = getattr(block, "is_error", False)
            label = "MCP error" if is_error else "MCP result"
            with st.expander(label):
                for item in getattr(block, "content", []) or []:
                    text = getattr(item, "text", None)
                    if text:
                        st.text(text)


def run_agentic_loop(client, model_id, api_messages, tools, mcp_servers,
                     mcp_tools, tmdb_tok, status, text_container):
    """Agentic loop with streaming text and parallel tool execution."""
    all_tools = tools + mcp_tools
    all_content = []
    full_text = ""

    for round_num in range(MAX_TOOL_ROUNDS):
        # On the last round, drop tools to force a final text answer
        is_last_round = round_num == MAX_TOOL_ROUNDS - 1
        kwargs = {
            "model": model_id,
            "max_tokens": 8192,
            "system": SYSTEM_PROMPT,
            "messages": api_messages,
            "thinking": {"type": "adaptive"},
        }
        if not is_last_round:
            kwargs["tools"] = all_tools

        if mcp_servers:
            # MCP path -- non-streaming (beta endpoint)
            kwargs["mcp_servers"] = mcp_servers
            kwargs["betas"] = ["mcp-client-2025-11-20"]
            response = client.beta.messages.create(**kwargs)
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    full_text += block.text
                    text_container.markdown(full_text)
        else:
            # Standard path -- streaming
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    full_text += text
                    text_container.markdown(full_text + "▌")
                response = stream.get_final_message()
            text_container.markdown(full_text)

        all_content.extend(response.content)

        tool_use_blocks = [
            b for b in response.content
            if getattr(b, "type", None) == "tool_use"
        ]

        if response.stop_reason == "end_turn" or not tool_use_blocks:
            break

        # Show what we're searching
        labels = [tool_call_label(tb) for tb in tool_use_blocks]
        status.update(
            label=f"Running {len(tool_use_blocks)} tool(s): "
                  f"{', '.join(labels)[:80]}..."
        )

        # Execute tools in parallel
        api_messages.append(
            {"role": "assistant", "content": response.content}
        )
        results = execute_tools_parallel(tool_use_blocks, tmdb_tok)
        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": results[tb.id],
            }
            for tb in tool_use_blocks
        ]
        api_messages.append({"role": "user", "content": tool_results})

        # Add spacing between rounds
        full_text += "\n\n"

    return all_content, full_text


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None

# ---------------------------------------------------------------------------
# Resume upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a resume / CV to research collaborators",
    type=["pdf", "txt"],
    help="Upload a PDF or text resume. Ask a question below to analyze it.",
)

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()
    file_name = uploaded_file.name
    if file_name.lower().endswith(".pdf"):
        st.session_state.uploaded_file_data = {
            "type": "pdf",
            "name": file_name,
            "b64": base64.standard_b64encode(raw_bytes).decode("utf-8"),
        }
    else:
        st.session_state.uploaded_file_data = {
            "type": "text",
            "name": file_name,
            "text": raw_bytes.decode("utf-8", errors="replace"),
        }
    st.success(f"Loaded **{file_name}** — ask a question below to analyze.")


def build_user_content(prompt_text):
    """Build the content array for a user message, including any uploaded file."""
    file_data = st.session_state.uploaded_file_data
    if file_data is None:
        return prompt_text  # plain string

    blocks = []
    if file_data["type"] == "pdf":
        blocks.append({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": file_data["b64"],
            },
            "title": file_data["name"],
        })
    else:
        blocks.append({
            "type": "text",
            "text": f"--- RESUME: {file_data['name']} ---\n{file_data['text']}\n--- END RESUME ---",
        })
    blocks.append({"type": "text", "text": prompt_text})
    return blocks


# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["display"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if prompt := st.chat_input(
    "Research TV shows, cast, ratings, production details..."
):
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    # What the user sees in the chat
    display_text = prompt
    if st.session_state.uploaded_file_data:
        fname = st.session_state.uploaded_file_data["name"]
        display_text = f"[Attached: {fname}]\n\n{prompt}"

    st.session_state.messages.append({
        "role": "user",
        "display": display_text,
        "content": build_user_content(prompt),
    })
    with st.chat_message("user"):
        st.markdown(display_text)

    # Clear file after sending so it's not re-attached on every message
    st.session_state.uploaded_file_data = None

    client = anthropic.Anthropic(api_key=api_key)
    tools, mcp_servers, mcp_tools = build_tools_and_mcp()
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        try:
            # Text streams into this container
            text_container = st.empty()

            with st.status("Researching...", expanded=True) as status:
                all_content, full_text = run_agentic_loop(
                    client, model, api_messages, tools, mcp_servers,
                    mcp_tools, tmdb_token, status, text_container,
                )
                status.update(
                    label="Research complete", state="complete",
                    expanded=False,
                )

            # Show tool call details below the response
            display_tool_calls(all_content)

            st.session_state.messages.append({
                "role": "assistant",
                "display": full_text.strip(),
                "content": full_text.strip(),
            })

        except anthropic.BadRequestError as e:
            st.error(f"Bad request: {e.message}")
        except anthropic.AuthenticationError:
            st.error(
                "Invalid API key. Please check your key in the sidebar."
            )
        except anthropic.RateLimitError:
            st.error("Rate limited. Please wait a moment and try again.")
        except anthropic.APIStatusError as e:
            st.error(f"API error ({e.status_code}): {e.message}")
        except anthropic.APIConnectionError:
            st.error("Connection error. Check your internet connection.")
