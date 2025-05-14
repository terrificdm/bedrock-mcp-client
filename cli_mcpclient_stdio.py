import asyncio
import sys
import boto3
import json

from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class Message:
    role: str
    content: List[Dict[str, Any]]

    @classmethod
    def user(cls, text: str) -> 'Message':
        return cls(role="user", content=[{"text": text}])

    @classmethod
    def assistant(cls, text: str) -> 'Message':
        return cls(role="assistant", content=[{"text": text}])

    @classmethod
    def tool_result(cls, tool_use_id: str, content: dict) -> 'Message':
        return cls(
            role="user",
            content=[{
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"json": {"text": content[0].text}}]
                }
            }]
        )

    @classmethod
    def tool_request(cls, tool_use_id: str, name: str, input_data: dict) -> 'Message':
        return cls(
            role="assistant",
            content=[{
                "toolUse": {
                    "toolUseId": tool_use_id,
                    "name": name,
                    "input": input_data
                }
            }]
        )

    @staticmethod
    def to_bedrock_format(tools_list: List[Dict]) -> List[Dict]:
        return [{
            "toolSpec": {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": tool["input_schema"]["properties"],
                        "required": tool["input_schema"]["required"]
                    }
                }
            }
        } for tool in tools_list]
        
class MCPClient:
    # Default configuration, can be overridden during initialization
    DEFAULT_CONFIG = {
        "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "region_name": "us-east-1",
        "max_tokens": 1000,
        "temperature": 0
    }
    
    # Maximum number of tool call turns in a single conversation
    MAX_TOOL_TURNS = 10
    
    def __init__(self, **kwargs):
        # Dictionary to store multiple server sessions
        self.servers = {}
        self.exit_stack = AsyncExitStack()
        
        # Tool to server mapping and tools cache
        self.tool_server_map = {}
        self.cached_tools = []
        
        # Merge default config with user-provided config
        self.config = {**self.DEFAULT_CONFIG, **kwargs}
        
        # Check AWS credentials
        self._check_aws_credentials()
        
        # Initialize Bedrock client
        self.bedrock = boto3.client(
            service_name='bedrock-runtime', 
            region_name=self.config["region_name"]
        )
        
        # Initialize conversation history
        self.conversation_history = []
        
    def _check_aws_credentials(self):
        """Check if AWS credentials are configured"""
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials is None:
                print("\nWARNING: No AWS credentials found. Please ensure AWS credentials are configured.")
                print("You can configure credentials by:")
                print("1. Setting environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
                print("2. Using AWS CLI by running 'aws configure'")
                print("3. Configuring credentials in ~/.aws/credentials file")
        except Exception as e:
            print(f"\nWARNING: Error checking AWS credentials: {str(e)}")

    async def connect_to_servers(self, config_path: str = "mcp_config.json"):
        """
        Connect to multiple MCP servers using configuration from a JSON file
        
        Args:
            config_path: Path to the JSON configuration file
        """
        try:
            # Load configuration from JSON file
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if "mcpServers" not in config:
                raise ValueError(f"Invalid configuration file: 'mcpServers' section not found in {config_path}")
            
            # Check if any servers are defined
            if not config["mcpServers"]:
                raise ValueError(f"No servers defined in configuration file {config_path}")
            
            # Connect to each server defined in the config
            for server_name, server_config in config["mcpServers"].items():
                # Validate server configuration
                if "command" not in server_config or "args" not in server_config:
                    print(f"WARNING: Invalid server configuration for '{server_name}': must contain 'command' and 'args'. Skipping.")
                    continue
                
                try:
                    # Create server parameters
                    server_params = StdioServerParameters(
                        command=server_config["command"],
                        args=server_config["args"],
                        env=server_config.get("env", None)
                    )
                    
                    # Connect to the server
                    stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                    stdio, write = stdio_transport
                    session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                    await session.initialize()
                    
                    # Store the session in our servers dictionary
                    self.servers[server_name] = {
                        "session": session,
                        "stdio": stdio,
                        "write": write
                    }
                
                except Exception as e:
                    print(f"ERROR: Failed to connect to server '{server_name}': {str(e)}")
            
            # Check if we connected to any servers
            if not self.servers:
                raise Exception("Failed to connect to any MCP servers")
            
            # Get all tools from all servers at once and cache them
            available_tools = []
            tools_by_server = {}
            
            for server_name, server_data in self.servers.items():
                try:
                    response = await server_data["session"].list_tools()
                    tool_names = [tool.name for tool in response.tools]
                    tools_by_server[server_name] = tool_names
                    
                    # Also build the full tools list for caching
                    server_tools = [{
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                        "server_name": server_name
                    } for tool in response.tools]
                    available_tools.extend(server_tools)
                except Exception as e:
                    print(f"ERROR: Failed to get tools from server '{server_name}': {str(e)}")
            
            # Update the tool-to-server mapping and cache the tools
            self.tool_server_map = {tool["name"]: tool["server_name"] for tool in available_tools}
            self.cached_tools = available_tools
            
            # Print a summary of all connected servers and their tools
            print("\nConnected to MCP servers with the following tools:")
            for server_name, tools in tools_by_server.items():
                print(f"  - {server_name}: {tools}")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in configuration file: {config_path}")
        except Exception as e:
            raise Exception(f"Failed to connect to servers: {str(e)}")

    async def cleanup(self):
        """Clean up all server connections"""
        # Clear the servers dictionary
        self.servers = {}
        # Close all connections
        await self.exit_stack.aclose()
    
    def _make_bedrock_stream_request(self, messages: List[Dict], tools: List[Dict]):
        try:
            return self.bedrock.converse_stream(
                modelId=self.config["model_id"],
                messages=messages,
                inferenceConfig={
                    "maxTokens": self.config["max_tokens"], 
                    "temperature": self.config["temperature"]
                },
                toolConfig={"tools": tools}
            )
        except boto3.exceptions.Boto3Error as e:
            print(f"\nERROR: AWS Bedrock request failed: {str(e)}")
            print("Possible reasons:")
            print("1. Invalid or expired AWS credentials")
            print("2. Selected model is not supported in the current region")
            print("3. You don't have permission to use Bedrock")
            print("4. Network connectivity issues")
            print("\nPlease check your AWS configuration and try again.")
            raise
        except Exception as e:
            print(f"\nERROR: Bedrock stream request failed: {str(e)}")
            if messages:
                print(f"First message: {json.dumps(messages[0])[:200]}...")
            raise

    async def get_all_tools(self, refresh=False):
        """
        Get all available tools from all connected servers
        
        Args:
            refresh: Whether to refresh the cached tools
            
        Returns:
            List of available tools with server information
        """
        # If we already have tools cached and don't need to refresh, return them
        if hasattr(self, 'cached_tools') and self.cached_tools and not refresh:
            return self.cached_tools
            
        # If refresh is requested, actually fetch the tools
        if refresh:
            available_tools = []
            for server_name, server_data in self.servers.items():
                try:
                    response = await server_data["session"].list_tools()
                    server_tools = [{
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                        "server_name": server_name  # Add server name to track which server owns this tool
                    } for tool in response.tools]
                    available_tools.extend(server_tools)
                except Exception as e:
                    print(f"\nERROR: Failed to get tools from server '{server_name}': {str(e)}")
            
            # Update the tool-to-server mapping
            self.tool_server_map = {tool["name"]: tool["server_name"] for tool in available_tools}
            
            # Cache the tools for future use
            self.cached_tools = available_tools
        
        return self.cached_tools
    
    async def process_query(self, query: str):
        """
        Process user query and generate response
        
        Args:
            query: User input query text
        """
        # Check if query is empty or contains only whitespace
        if not query or query.isspace():
            print("\nERROR: Empty queries are not allowed. Please enter a valid query.")
            return
            
        # Add user message to conversation history
        user_message = Message.user(query).__dict__
        self.conversation_history.append(user_message)
        
        # Use the cached tools - no need to refresh on every query
        available_tools = self.cached_tools
        
        if not available_tools:
            print("\nERROR: No tools available from any connected servers.")
            error_message = {
                "role": "assistant", 
                "content": [{"text": "No tools are currently available. Please check server connections."}]
            }
            self.conversation_history.append(error_message)
            return

        # Format tools for Bedrock (without the server_name field)
        bedrock_tools = Message.to_bedrock_format([{k: v for k, v in tool.items() if k != 'server_name'} 
                                                 for tool in available_tools])

        # Process the response with streaming
        await self._process_stream_response(self.conversation_history, bedrock_tools)
        
    async def _process_stream_response(self, messages: List[Dict], bedrock_tools: List[Dict]):
        """
        Process streaming response and handle tool calls
        
        Args:
            messages: List of conversation history messages
            bedrock_tools: Formatted Bedrock tools list
            
        Returns:
            None
        """
        turn_count = 0

        while turn_count < self.MAX_TOOL_TURNS:
            try:
                # Make streaming request
                response = self._make_bedrock_stream_request(messages, bedrock_tools)
                
                # Process the streaming response
                stop_reason, message = await self._handle_stream_response(response)
                
                # Add the message to conversation history
                messages.append(message)
                
                if stop_reason == "tool_use":
                    # Handle tool use
                    for content in message['content']:
                        if 'toolUse' in content:
                            tool_info = content['toolUse']
                            tool_result = await self._handle_tool_call(tool_info)
                            messages.append(tool_result)
                    
                    turn_count += 1
                    # Continue the conversation with the tool result
                    continue
                else:
                    # End of conversation for this turn
                    break
            except Exception as e:
                print(f"\nERROR in stream response processing: {str(e)}")
                # Add error message to conversation history
                error_message = {
                    "role": "assistant", 
                    "content": [{"text": f"An error occurred: {str(e)}. Please try again."}]
                }
                messages.append(error_message)
                break

    async def _handle_stream_response(self, response) -> tuple[str, dict]:
        """
        Process streaming response from Bedrock
        
        Args:
            response: The streaming response from Bedrock
            
        Returns:
            tuple: (stop_reason, message) where stop_reason is a string indicating why the 
                  response stopped and message is the complete message dictionary
        """
        message = {}
        content = []
        message['content'] = content
        text = ''
        tool_use = {}
        stop_reason = ""

        try:
            # Stream the response and build the message
            for chunk in response['stream']:
                if 'messageStart' in chunk:
                    message['role'] = chunk['messageStart']['role']
                elif 'contentBlockStart' in chunk:
                    if 'toolUse' in chunk['contentBlockStart'].get('start', {}):
                        tool = chunk['contentBlockStart']['start']['toolUse']
                        tool_use['toolUseId'] = tool['toolUseId']
                        tool_use['name'] = tool['name']
                elif 'contentBlockDelta' in chunk:
                    delta = chunk['contentBlockDelta']['delta']
                    if 'toolUse' in delta:
                        if 'input' not in tool_use:
                            tool_use['input'] = ''
                        tool_use['input'] += delta['toolUse']['input']
                    elif 'text' in delta:
                        text += delta['text']
                        print(delta['text'], end='', flush=True)  # Stream output character by character
                elif 'contentBlockStop' in chunk:
                    if 'input' in tool_use:
                        try:
                            tool_use['input'] = json.loads(tool_use['input'])
                        except json.JSONDecodeError:
                            # Handle case where tool input is not valid JSON
                            print(f"\nWARNING: Tool input is not valid JSON: {tool_use['input'][:100]}...")
                            # Try to convert to a valid JSON object instead of keeping as string
                            try:
                                # Create a proper JSON object with the string as a value
                                tool_use['input'] = {"text": tool_use['input']}
                                print("Converted invalid JSON to a valid JSON object")
                            except Exception as e:
                                print(f"Failed to convert to JSON object: {str(e)}")
                                # As last resort, keep original string format
                                print("Will keep original string format")
                        content.append({'toolUse': tool_use})
                        tool_use = {}
                    else:
                        content.append({'text': text})
                        text = ''
                elif 'messageStop' in chunk:
                    stop_reason = chunk['messageStop']['stopReason']
                    
            # Handle any remaining text that wasn't added to content
            if text and not any(item.get('text') == text for item in content):
                content.append({'text': text})
                
            return stop_reason, message
            
        except KeyError as e:
            print(f"\nERROR: Missing key in response chunk: {str(e)}")
            # Return a partial message with error information
            if 'role' not in message:
                message['role'] = 'assistant'
            content.append({'text': f"Error processing response: Missing key {str(e)}"})
            return "error", message
            
        except Exception as e:
            print(f"\nERROR during stream processing: {str(e)}")
            # Return a message with error information
            return "error", {"role": "assistant", "content": [{"text": f"Error processing response: {str(e)}"}]}

    async def _handle_tool_call(self, tool_info: Dict) -> Dict:
        """
        Handle a tool call request
        
        Args:
            tool_info: Dictionary containing tool information including name, input, and toolUseId
            
        Returns:
            Dict: Tool result message in the format expected by Bedrock
        """
        try:
            tool_name = tool_info['name']
            tool_args = tool_info['input']
            tool_use_id = tool_info['toolUseId']

            # Find which server has this tool
            server_name = self.tool_server_map.get(tool_name)
            if not server_name or server_name not in self.servers:
                error_msg = f"Tool '{tool_name}' is not available on any connected server"
                print(f"\nERROR: {error_msg}")
                return {
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"json": {"text": f"Error: {error_msg}"}}]
                        }
                    }]
                }

            print(f"\n[Calling tool {tool_name} on server {server_name} with args {json.dumps(tool_args)}]")

            # Get the session for the server that has this tool
            session = self.servers[server_name]["session"]
            
            # Call the tool through the appropriate MCP server
            result = await session.call_tool(tool_name, tool_args)

            # Create tool result message
            tool_result_message = {
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"json": {"text": result.content[0].text}}]
                    }
                }]
            }

            return tool_result_message
            
        except KeyError as e:
            print(f"\nERROR: Missing key in tool info: {str(e)}")
            # Return an error message as tool result
            return {
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": tool_info.get('toolUseId', 'unknown'),
                        "content": [{"json": {"text": f"Error: Missing required tool information - {str(e)}"}}]
                    }
                }]
            }
            
        except Exception as e:
            print(f"\nERROR during tool call: {str(e)}")
            # Return an error message as tool result
            return {
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": tool_info.get('toolUseId', 'unknown'),
                        "content": [{"json": {"text": f"Error calling tool: {str(e)}"}}]
                    }
                }]
            }
        
    async def chat_loop(self):
        """Run interactive chat loop, process user input and generate responses"""
        print("\nMCP Client Started!\nType queries or 'quit' to exit, 'clear' to clear history, 'refresh' to refresh the tools cache, 'help' to show more information.")
        print("Conversation history is maintained between queries.")
        
        try:
            while True:
                try:
                    query = input("\nQuery: ").strip()
                    if query.lower() == 'quit':
                        break
                    elif query.lower() == 'clear':
                        self.conversation_history = []
                        print("Conversation history cleared.")
                        continue
                    elif query.lower() == 'help':
                        print("\nAvailable commands:")
                        print("  quit  - Exit the program")
                        print("  clear - Clear conversation history")
                        print("  help  - Show this help information")
                        print("  refresh - Refresh the tools cache")
                        continue
                    elif query.lower() == 'refresh':
                        await self.get_all_tools(refresh=True)
                        print("Tools cache refreshed.")
                        continue
                    elif not query:
                        print("ERROR: Empty queries are not allowed. Please enter a valid query.")
                        continue
                        
                    await self.process_query(query)
                    print()  # Add a newline after response
                except KeyboardInterrupt:
                    print("\n\nKeyboard interrupt detected. Type 'quit' to exit or continue with a new query.")
                    continue
                except Exception as e:
                    print(f"\nERROR: {str(e)}")
                    print("\nContinuing with a new query. Type 'quit' to exit.")
        finally:
            # Ensure resources are properly cleaned up even if an unhandled exception occurs
            print("\nCleaning up resources...")
            try:
                await self.cleanup()
                print("Resources cleaned up successfully.")
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
                
async def main():
    """Main function, start the client using configuration from JSON file"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MCP Client for AWS Bedrock')
    parser.add_argument('--config', '-c', default='mcp_config.json', help='Path to configuration file (default: mcp_config.json)')
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
        client = MCPClient(**config)
        await client.connect_to_servers(args.config)
        await client.chat_loop()
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
        # Only attempt cleanup if client was initialized
        if client is not None:
            try:
                await client.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())