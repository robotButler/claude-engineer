CLAUDE_TOOLS = [
    {
        "name": "create_folder",
        "description": "Create a new folder at the specified path. Use this when you need to create a new directory in the project structure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path where the folder should be created"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "create_file",
        "description": "Create a new file at the specified path with optional content. Use this when you need to create a new file in the project structure.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path where the file should be created"
                },
                "content": {
                    "type": "string",
                    "description": "The initial content of the file (optional)"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "apply_patch",
        "description": "Apply a patch to a file or directory. Use this when you need to make changes to an existing file or directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pwd": {
                    "type": "string",
                    "description": "The working directory where the patch should be applied"
                },
                "patch": {
                    "type": "string",
                    "description": "The patch to apply"
                }
            },
            "required": ["pwd", "patch"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file at the specified path and prepend line numbers on each line. Use this when you need to examine the contents of an existing file. Use the line numbers when writing a patch, ignore them otherwise.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the file to read"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_files",
        "description": "List all files and directories in the root folder where the script is running. Use this when you need to see the contents of the current directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path of the folder to list (default: current directory)"
                }
            }
        }
    },
    {
        "name": "tavily_search",
        "description": "Perform a web search using Tavily API to get up-to-date information or additional context. Use this when you need current information or feel a search could provide a better answer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "run_build_command",
        "description": "Run a build command in the current directory. Use this when you need to build a project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pwd": {
                    "type": "string",
                    "description": "The working directory where the build command should be run"
                },
                "command": {
                    "type": "string",
                    "description": "The build command to run"
                }
            },
            "required": ["pwd", "command"]
        }
    },
    {
        "name": "end_automode",
        "description": "End the automode and return to manual mode. Use this when you know your goals are completed and the build passes (if the user provided a build command).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]