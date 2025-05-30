import asyncio
import sys
import json
import io
import argparse
from typing import Dict, Any, List, Optional, Tuple

from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

# Set standard input/output encoding to UTF-8
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class StrandsMCPClient:
    # Default configuration, can be overridden during initialization
    DEFAULT_CONFIG = {
        "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "region_name": "us-east-1",
        "max_tokens": 1000,
        "temperature": 0
    }
    
    # Default configuration path
    DEFAULT_CONFIG_PATH = "mcp_config.json"
    
    def __init__(self, **kwargs):
        # Merge default config with user-provided config
        self.config = {**self.DEFAULT_CONFIG, **kwargs}
        
        # Initialize Bedrock model
        self.bedrock_model = BedrockModel(
            model_id=self.config["model_id"],
            region_name=self.config["region_name"],
            max_tokens=self.config["max_tokens"],
            temperature=self.config["temperature"]
        )
        
        # Initialize agent with no tools yet (will add after connecting to servers)
        self.agent = None
        
        # Track all available tools
        self.all_tools = []
        
        # Store MCP clients to keep sessions active
        self.mcp_clients = {}
        
        # Store configuration path for reuse
        self.config_path = self.DEFAULT_CONFIG_PATH
    
    def _log_error(self, message: str) -> None:
        """Print formatted error message"""
        print(f"\nERROR: {message}")
    
    def _log_warning(self, message: str) -> None:
        """Print formatted warning message"""
        print(f"\nWARNING: {message}")
    
    async def _load_config(self, config_path: str) -> Dict:
        """
        Load and validate configuration from a JSON file
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            Dict: Loaded configuration
            
        Raises:
            FileNotFoundError: If configuration file is not found
            ValueError: If configuration is invalid
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if "mcpServers" not in config:
                raise ValueError(f"Invalid configuration file: 'mcpServers' section not found in {config_path}")
            
            if not config["mcpServers"]:
                raise ValueError(f"No servers defined in configuration file {config_path}")
                
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file: {config_path}")
    
    async def _setup_servers_and_tools(self, config: Dict, is_refresh: bool = False) -> None:
        """
        Connect to servers and set up tools
        
        Args:
            config: Configuration dictionary with mcpServers section
            is_refresh: Whether this is a refresh operation
            
        Raises:
            Exception: If failed to get tools from any MCP servers
        """
        # Clean up any existing clients
        await self._cleanup_clients()
        
        # Connect to each server defined in the config
        all_tools = []
        tools_by_server = {}
        
        for server_name, server_config in config["mcpServers"].items():
            tools, success = await self._connect_to_server(server_name, server_config)
            if success:
                tools_by_server[server_name] = [tool.tool_name for tool in tools]
                all_tools.extend(tools)
        
        # Check if we got any tools
        if not all_tools:
            message = "Failed to get tools from any MCP servers"
            if is_refresh:
                self._log_error(message)
                return
            else:
                raise Exception(message)
        
        # Store all tools
        self.all_tools = all_tools
        
        # Create or update agent with tools
        if is_refresh and self.agent:
            self.agent = Agent(
                model=self.bedrock_model,
                tools=all_tools,
                messages=self.agent.messages
            )
        else:
            self.agent = Agent(
                model=self.bedrock_model,
                tools=all_tools
            )
        
        # Print a summary of all connected servers and their tools
        status = "Refreshed" if is_refresh else "Connected to"
        print(f"\n{status} MCP servers with the following tools:")
        for server_name, tools in tools_by_server.items():
            print(f"  - {server_name}: {tools}")
        
    async def connect_to_servers(self, config_path: str = None):
        """
        Connect to multiple MCP servers using configuration from a JSON file
        
        Args:
            config_path: Path to the JSON configuration file
        """
        # Use provided config_path or fall back to default
        config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config_path = config_path
        
        try:
            config = await self._load_config(config_path)
            await self._setup_servers_and_tools(config)
        except Exception as e:
            raise Exception(f"Failed to connect to servers: {str(e)}")

    async def _cleanup_clients(self):
        """Clean up MCP client connections"""
        for client in self.mcp_clients.values():
            try:
                client.__exit__(None, None, None)  # Manually exit context
            except Exception as e:
                self._log_warning(f"Error closing client: {str(e)}")
        self.mcp_clients = {}
    
    async def _connect_to_server(self, server_name, server_config) -> Tuple[List, bool]:
        """
        Connect to a single MCP server and get its tools
        
        Args:
            server_name: Name of the server
            server_config: Server configuration dictionary
            
        Returns:
            tuple: (tools_list, success_flag)
        """
        if "command" not in server_config or "args" not in server_config:
            self._log_warning(f"Invalid server configuration for '{server_name}': must contain 'command' and 'args'. Skipping.")
            return [], False
            
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config["args"],
                env=server_config.get("env", None)
            )
            
            # Use context manager to initialize the client but keep the reference
            mcp_client = MCPClient(lambda: stdio_client(server_params))
            mcp_client.__enter__()  # Manually enter context to initialize session
            
            # Store the client to keep the session active
            self.mcp_clients[server_name] = mcp_client
            
            # Get tools from this server using the synchronous method
            tools = mcp_client.list_tools_sync()
            return tools, True
            
        except Exception as e:
            self._log_error(f"Failed to connect to server '{server_name}': {str(e)}")
            return [], False
    
    async def refresh_tools(self, config_path: str = None):
        """
        Refresh the tools from all connected servers
        
        Args:
            config_path: Path to the JSON configuration file (optional)
        """
        # Use provided config_path or fall back to stored path
        config_path = config_path or self.config_path
        
        try:
            config = await self._load_config(config_path)
            await self._setup_servers_and_tools(config, is_refresh=True)
        except FileNotFoundError:
            self._log_error(f"Configuration file not found: {config_path}")
        except ValueError as e:
            self._log_error(str(e))
        except Exception as e:
            self._log_error(f"Failed to refresh tools: {str(e)}")
    
    async def process_query(self, query: str):
        """
        Process user query and generate response
        
        Args:
            query: User input query text
        """
        # Check if query is empty or contains only whitespace
        if not query or query.isspace():
            self._log_error("Empty queries are not allowed. Please enter a valid query.")
            return
        
        if not self.agent:
            self._log_error("Agent not initialized. Please connect to servers first.")
            return
            
        if not self.all_tools:
            self._log_error("No tools available. Please refresh tools or check server connections.")
            return
        
        # Process the query with the agent
        # Define a custom callback handler to stream the output
        def callback_handler(**kwargs):
            if "data" in kwargs:
                print(kwargs["data"], end="", flush=True)
        
        # Use the agent to process the query
        try:
            self.agent(query, callback_handler=callback_handler)
        except Exception as e:
            self._log_error(f"Failed to process query: {str(e)}")
    
    async def chat_loop(self, config_path: str = None):
        """
        Run interactive chat loop, process user input and generate responses
        
        Args:
            config_path: Path to the JSON configuration file
        """
        # Update stored config path if provided
        if config_path:
            self.config_path = config_path
        
        print("\nMCP Client Started!\nType queries or 'quit' to exit, 'clear' to clear history, 'refresh' to refresh the tools cache, 'help' to show more information.")
        print("Conversation history is maintained between queries.")
        
        try:
            while True:
                try:
                    query = input("\nQuery: ").strip()
                    if query.lower() == 'quit':
                        break
                    elif query.lower() == 'clear':
                        if self.agent:
                            # Clear conversation history
                            self.agent = Agent(
                                model=self.bedrock_model,
                                tools=self.all_tools
                            )
                        print("\nConversation history cleared.")
                        continue
                    elif query.lower() == 'help':
                        print("\nAvailable commands:")
                        print("  quit  - Exit the program")
                        print("  clear - Clear conversation history")
                        print("  help  - Show this help information")
                        print("  refresh - Refresh the tools cache")
                        continue
                    elif query.lower() == 'refresh':
                        await self.refresh_tools()
                        continue  # refresh_tools already prints a message
                    elif not query:
                        self._log_error("Empty queries are not allowed. Please enter a valid query.")
                        continue
                        
                    await self.process_query(query)
                    print()  # Add a newline after response
                except KeyboardInterrupt:
                    print("\n\nKeyboard interrupt detected. Type 'quit' to exit or continue with a new query.")
                    continue
                except Exception as e:
                    self._log_error(str(e))
                    print("\nContinuing with a new query. Type 'quit' to exit.")
        finally:
            print("\nCleaning up resources...")
            await self._cleanup_clients()

async def main():
    """Main function, start the client using configuration from JSON file"""
    parser = argparse.ArgumentParser(description='MCP Client for AWS Bedrock using Strands Agents')
    parser.add_argument('--config', '-c', default=StrandsMCPClient.DEFAULT_CONFIG_PATH, help=f'Path to configuration file (default: {StrandsMCPClient.DEFAULT_CONFIG_PATH})')
    parser.add_argument('--region', '-r', help='AWS region name (default: us-east-1)')
    parser.add_argument('--model', '-m', help='Bedrock model ID (default: anthropic.claude-3-sonnet-20240229-v1:0)')
    parser.add_argument('--max-tokens', '-t', type=int, help='Maximum tokens to generate (default: 1000)')
    parser.add_argument('--temperature', '-T', type=float, help='Generation temperature (default: 0)')
    
    args = parser.parse_args()
    
    # Build config dictionary, only include non-None values
    config = {k: v for k, v in {
        'region_name': args.region,
        'model_id': args.model,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature
    }.items() if v is not None}
    
    client = None
    try:
        client = StrandsMCPClient(**config)
        await client.connect_to_servers(args.config)
        await client.chat_loop(args.config)
    except FileNotFoundError as e:
        print(f"\nERROR: Configuration file not found: {str(e)}")
        print("Please check the path and try again.")
        sys.exit(1)
    except ValueError as e:
        print(f"\nERROR: Invalid argument or configuration: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {str(e)}")
        print("If this issue persists, please check your configuration, AWS credentials, and network connection.")
        sys.exit(1)
    finally:
        # Clean up resources if client was initialized
        if client is not None:
            try:
                await client._cleanup_clients()
            except Exception as e:
                print(f"\nError during cleanup: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())