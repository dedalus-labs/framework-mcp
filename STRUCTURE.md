# Project Structure

## Essential Files (Required for Dedalus)
```
.
├── requirements.txt      # Python dependencies (Dedalus installs these)
├── src/
│   └── server.py        # Main MCP server (Dedalus runs this)
└── docs/               # Your documentation files to serve
```

## Full Structure
```
.
├── README.md           # Quick start guide
├── requirements.txt    # Python dependencies
├── LICENSE            # MIT license
├── src/
│   └── server.py      # Main MCP server
├── docs/
│   ├── guides/        # Technical guides
│   │   ├── getting-started.md
│   │   ├── advanced-tools.md
│   │   ├── agent-handoffs.md
│   │   └── deployment.md
│   └── hackathon/     # Hackathon information
│       ├── yc-agents-hackathon.md
│       ├── hackathon-rules.md
│       └── add-your-ideas.md
├── examples/          # Example code
│   ├── client.py      # Dedalus client example
│   ├── workflows.py   # Advanced workflows
│   ├── Dockerfile     # Container example
│   └── requirements.txt
├── tests/
│   └── test_server.py # Server tests
├── config/
│   └── .env.example   # Environment template
└── scripts/
    └── setup.sh       # Local setup script
```

## Deployment

Dedalus only needs:
1. `requirements.txt` - to install dependencies
2. `src/server.py` - to run your MCP server
3. `docs/` - your documentation files

Everything else is for local development and examples.