import streamlit as st
import json
import threading
import asyncio
import atexit
import time
from typing import Dict, List, Any, Optional, Tuple

from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

# Constants
DEFAULT_CONFIG_PATH = "mcp_config.json"
DEFAULT_MODEL = "Anthropic Claude-3.5-haiku"
DEFAULT_REGION = "us-east-1"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.0

# Set page configuration
st.set_page_config(
    page_title="MCP Client for Bedrock",
    page_icon="./utils/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define event loop manager class
class AsyncioEventLoopThread:
    """Run event loop in a separate thread"""
    
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
    
    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def run_coroutine(self, coroutine):
        """Run coroutine in the event loop and return result"""
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return future.result()
    
    def close(self):
        """Close the event loop"""
        try:
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join(timeout=5)  # Set timeout to avoid infinite waiting
                if self.loop and not self.loop.is_closed():
                    self.loop.close()
        except Exception as e:
            print(f"Error closing event loop: {e}")

# Define StreamlitHandler for handling streaming responses
class StreamlitHandler:
    """Handler for streaming responses to Streamlit UI"""
    
    def __init__(self, placeholder, update_interval=0.05):
        """
        Initialize the handler with a Streamlit placeholder
        
        Args:
            placeholder: Streamlit placeholder to update with streaming content
            update_interval: Interval in seconds between UI updates
        """
        self.placeholder = placeholder
        self.update_interval = update_interval
        self.message_container = ""
        self.last_update_time = 0
        self.tool_calls = set()
        self.has_tool_call = False
    
    def __call__(self, **kwargs):
        """
        Callback function for handling streaming events
        
        Args:
            **kwargs: Callback arguments from Strands agent
        """
        current_time = time.time()
        
        # Handle text streaming
        if "data" in kwargs:
            # Stream text output
            self.message_container += kwargs["data"]
            
            # Only update UI at specified intervals to avoid excessive refreshes
            if current_time - self.last_update_time >= self.update_interval:
                self.placeholder.markdown(self.message_container)
                self.last_update_time = current_time
        
        # Handle tool calls
        elif "current_tool_use" in kwargs and kwargs["current_tool_use"].get("name"):
            # Track tool usage
            tool_name = kwargs["current_tool_use"]["name"]
            server_name = st.session_state.tool_server_map.get(tool_name, "unknown")
            tool_call_message = f"Calling tool {tool_name} on server {server_name}"
            
            # Mark that we've had a tool call
            self.has_tool_call = True
            
            # Add tool call to conversation for display if not already added
            if tool_call_message not in self.tool_calls:
                self.tool_calls.add(tool_call_message)
                
                # Add tool call to UI only (not to conversation history yet)
                st.info(tool_call_message)

# Initialize session state
default_session_state = {
    'conversation': [],
    'connected': False,
    'mcp_servers': {},
    'tools': [],
    'tool_server_map': {},
    'server_tools_map': {},
    'current_config_path': "",
    'current_region': "",
    'current_model_id': DEFAULT_MODEL,
    'agent': None,
    'processing': False
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialize event loop
if 'asyncio_loop' not in st.session_state:
    st.session_state.asyncio_loop = AsyncioEventLoopThread()
    
    # Register cleanup function for program exit
    def cleanup_resources():
        if 'asyncio_loop' in st.session_state:
            st.session_state.asyncio_loop.close()
    
    atexit.register(cleanup_resources)

# Sidebar configuration
with st.sidebar:
    col1, col2 = st.columns([1, 3.5])
    with col1:
        st.image('./utils/logo.png')
    with col2:
        st.title("Bedrock MCP Client")

# Configuration file input
config_path = st.sidebar.text_input("MCP Configuration Path", value=DEFAULT_CONFIG_PATH)

# Update the current configuration path
st.session_state.current_config_path = config_path

# Model selection - using dropdown menu
model_options = ["Anthropic Claude-3.7-Sonnet", "Anthropic Claude-3.5-haiku", "Anthropic Claude-3.5-Sonnet-v1", "Anthropic Claude-3.5-Sonnet-v2"]
new_model_id = st.sidebar.selectbox("Bedrock Model", options=model_options, index=1)

# Detect model changes
if new_model_id != st.session_state.current_model_id:
    st.session_state.current_model_id = new_model_id

# Model ID mapping
model_id_mapping = {
    'Anthropic Claude-3.7-Sonnet': 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
    'Anthropic Claude-3.5-haiku': 'us.anthropic.claude-3-5-haiku-20241022-v1:0',
    'Anthropic Claude-3.5-Sonnet-v1': 'us.anthropic.claude-3-5-sonnet-20240620-v1:0',
    'Anthropic Claude-3.5-Sonnet-v2': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
}

# Get the actual model ID
model_id = model_id_mapping.get(new_model_id)

# AWS Region selection - using dropdown menu
region_options = ["us-east-1", "us-west-2"]
region_name = st.sidebar.selectbox("AWS Region", options=region_options, 
                                  index=region_options.index(DEFAULT_REGION) if DEFAULT_REGION in region_options else 0)

# Detect region changes
region_changed = st.session_state.current_region != region_name and st.session_state.current_region != ""

# Update current region
st.session_state.current_region = region_name

# Generation parameters
with st.sidebar.expander('Model Parameters', expanded=False):
    max_tokens = st.slider("Max tokens to output", min_value=100, 
                           max_value=(65536 if "claude-3-7-sonnet" in model_id 
                           else 4096), 
                           value=DEFAULT_MAX_TOKENS, step=100)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.1)

# MCP Server Manager class
class MCPServerManager:
    """Class to manage MCP server operations"""
    
    @staticmethod
    async def load_config(config_path: str) -> Dict:
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
    
    @staticmethod
    def start_servers(config_path: str) -> bool:
        """Start MCP servers from configuration file"""
        try:
            # Load configuration
            config = st.session_state.asyncio_loop.run_coroutine(
                MCPServerManager.load_config(config_path)
            )
            
            if "mcpServers" not in config or not config["mcpServers"]:
                MCPServerManager.handle_error("No servers defined in configuration file")
                return False
            
            # Start each server
            for server_name, server_config in config["mcpServers"].items():
                if "command" not in server_config or "args" not in server_config:
                    st.warning(f"Invalid server configuration for '{server_name}'")
                    continue
                
                # Store server config
                st.session_state.mcp_servers[server_name] = server_config
                
            return True
        except Exception as e:
            MCPServerManager.handle_error("Error loading server configuration", e)
            return False
    
    @staticmethod
    def handle_error(error_message: str, exception: Exception = None) -> None:
        """
        Centralized error handling function
        
        Args:
            error_message: The error message to display
            exception: Optional exception object
        """
        error_text = f"{error_message}"
        if exception:
            error_text += f": {str(exception)}"
        st.error(error_text)
        
    @staticmethod
    async def connect_to_server(server_name: str, server_config: Dict) -> Tuple[MCPClient, List, bool]:
        """
        Connect to a single MCP server and get its tools
        
        Args:
            server_name: Name of the server
            server_config: Server configuration dictionary
            
        Returns:
            tuple: (mcp_client, tools_list, success_flag)
        """
        if "command" not in server_config or "args" not in server_config:
            st.warning(f"Invalid server configuration for '{server_name}'")
            return None, [], False
            
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config["args"],
                env=server_config.get("env", None)
            )
            
            # Create MCP client
            mcp_client = MCPClient(lambda: stdio_client(server_params))
            mcp_client.__enter__()  # Manually enter context
            
            # Get tools from this server
            tools = mcp_client.list_tools_sync()
            
            # Add server name to tools for tracking
            for tool in tools:
                tool.server_name = server_name
                
            return mcp_client, tools, True
            
        except Exception as e:
            MCPServerManager.handle_error(f"Failed to connect to server '{server_name}'", e)
            return None, [], False
    
    @staticmethod
    def setup_servers_and_tools() -> bool:
        """Connect to servers and set up tools"""
        # Clean up any existing clients
        if hasattr(st.session_state, 'mcp_clients'):
            for client in st.session_state.mcp_clients.values():
                try:
                    client.__exit__(None, None, None)
                except Exception as e:
                    st.warning(f"Error closing client: {str(e)}")
        
        # Initialize clients dictionary
        st.session_state.mcp_clients = {}
        
        # Connect to each server
        all_tools = []
        tools_by_server = {}
        
        for server_name, server_config in st.session_state.mcp_servers.items():
            client, tools, success = st.session_state.asyncio_loop.run_coroutine(
                MCPServerManager.connect_to_server(server_name, server_config)
            )
            
            if success:
                # Store client
                st.session_state.mcp_clients[server_name] = client
                
                # Store tools by server
                tools_by_server[server_name] = [tool.tool_name for tool in tools]
                
                # Add tools to all tools list
                all_tools.extend(tools)
                
                # Update tool-server mapping
                for tool in tools:
                    st.session_state.tool_server_map[tool.tool_name] = server_name
        
        # Check if we got any tools
        if not all_tools:
            st.error("Failed to get tools from any MCP servers")
            return False
        
        # Store all tools and server tools map
        st.session_state.tools = all_tools
        st.session_state.server_tools_map = tools_by_server
        
        # Create Bedrock model
        bedrock_model = BedrockModel(
            model_id=model_id,
            region_name=region_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Create agent with tools
        st.session_state.agent = Agent(
            model=bedrock_model,
            tools=all_tools
        )
        
        return True

# Define a function to reconnect to MCP servers
def reconnect_mcp_servers(reason=""):
    with st.spinner(f"Reconnecting... {reason}"):
        # Reset connection status
        st.session_state.connected = False
        st.session_state.mcp_servers = {}
        st.session_state.tools = []
        st.session_state.tool_server_map = {}
        
        # Start MCP servers
        if MCPServerManager.start_servers(config_path):
            # Set up servers and tools
            if MCPServerManager.setup_servers_and_tools():
                st.session_state.connected = True
                return True
            else:
                st.error("Failed to set up servers and tools")
                return False
        else:
            st.error("Failed to start MCP servers")
            return False

# Auto-connect to MCP servers on app startup or refresh
if not st.session_state.connected:
    with st.spinner("Connecting to MCP servers..."):
        # Start MCP servers
        if MCPServerManager.start_servers(config_path):
            # Set up servers and tools
            if MCPServerManager.setup_servers_and_tools():
                st.session_state.connected = True
                st.success(f"Connected to {len(st.session_state.mcp_servers)} MCP servers with {len(st.session_state.tools)} tools, check left panel for details!")
            else:
                st.error("Failed to set up servers and tools")
        else:
            st.error("Failed to start MCP servers")

# If configuration path changes, trigger reconnection
if st.session_state.current_config_path != config_path and st.session_state.current_config_path != "":
    reconnect_mcp_servers("Configuration file changed")
    
# Function to update agent with new model settings
def update_agent_model(model_id, region_name, max_tokens, temperature):
    if st.session_state.connected and st.session_state.agent:
        with st.spinner(f"Updating model to {new_model_id} in region {region_name}..."):
            # Create new Bedrock model
            bedrock_model = BedrockModel(
                model_id=model_id,
                region_name=region_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Create new agent with updated model but keep the same tools and conversation
            st.session_state.agent = Agent(
                model=bedrock_model,
                tools=st.session_state.tools,
                messages=st.session_state.agent.messages if st.session_state.agent else None
            )
            
            st.success(f"Updated to model: {new_model_id} in region: {region_name}")

# If region changes or model changes, update agent
if region_changed or new_model_id != st.session_state.current_model_id:
    update_agent_model(model_id, region_name, max_tokens, temperature)

# Display MCP server connection info at the top of the chat page
if st.session_state.connected and hasattr(st.session_state, 'server_tools_map'):
    with st.sidebar.expander("MCP Connection Info", expanded=False):
        for server_name, tool_names in st.session_state.server_tools_map.items():
            selected_tool = st.selectbox(
                f"Tools for {server_name}",
                options=tool_names,
                key=f"tools_{server_name}"
            )

# Clear conversation button
if st.sidebar.button("New Conversation", type="primary"):
    # Create new agent with same model and tools but clear conversation
    if st.session_state.connected and st.session_state.agent:
        bedrock_model = st.session_state.agent.model
        st.session_state.agent = Agent(
            model=bedrock_model,
            tools=st.session_state.tools
        )
    st.session_state.conversation = []

# Display welcome message if conversation is empty
if not st.session_state.conversation:
    with st.chat_message("assistant", avatar="./utils/assistant.png"):
        st.markdown("I am an AI chatbot powered by Amazon Bedrock, what can I do for you？💬")

# Display conversation history
i = 0
while i < len(st.session_state.conversation):
    message = st.session_state.conversation[i]
    
    if message["role"] == "user":
        st.chat_message("user", avatar="./utils/user.png").markdown(message["content"])
        i += 1
    elif message["role"] == "assistant":
        with st.chat_message("assistant", avatar="./utils/assistant.png"):
            st.markdown(message["content"])
            
            # Check if next messages are tool calls and display them inside the same chat message
            j = i + 1
            while j < len(st.session_state.conversation) and st.session_state.conversation[j]["role"] == "tool_call":
                st.info(st.session_state.conversation[j]["content"])
                j += 1
            
            # Skip the tool call messages we just displayed
            i = j
    else:
        # Skip any standalone tool calls (should not happen with new logic)
        i += 1

# Initialize session state for processing flag if needed
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Process a message using Strands Agent
def process_message(user_input):
    if not st.session_state.connected or not st.session_state.agent:
        st.error("Please connect to MCP servers first")
        return
    
    # Set processing flag to prevent multiple submissions
    st.session_state.processing = True
    
    try:
        # Add user message to conversation for display
        st.session_state.conversation.append({
            "role": "user",
            "content": user_input
        })
        
        # Create a placeholder for the assistant's initial response
        with st.chat_message("assistant", avatar="./utils/assistant.png"):
            response_placeholder = st.empty()
        
        # Create stream handler with the placeholder
        stream_handler = StreamlitHandler(
            placeholder=response_placeholder,
            update_interval=0.05  # Update UI every 0.05 seconds
        )
        
        # Get the agent from session state
        agent = st.session_state.agent
        
        # Set the callback handler for this interaction
        agent.callback_handler = stream_handler
        
        # Process with agent - this will trigger streaming through the handler
        response = agent(user_input)
        
        # Ensure the final response is displayed correctly
        if not stream_handler.message_container:
            # If no streaming occurred, show the full response
            response_text = response.message["content"][0]["text"]
            response_placeholder.markdown(response_text)
            
            # Add to conversation history
            st.session_state.conversation.append({
                "role": "assistant",
                "content": response_text
            })
        else:
            # Ensure the final complete response is displayed
            response_placeholder.markdown(stream_handler.message_container)
            
            # Add the complete assistant response to conversation history first
            st.session_state.conversation.append({
                "role": "assistant",
                "content": stream_handler.message_container
            })
            
            # Add all tool calls to conversation history after the assistant response
            for tool_call in stream_handler.tool_calls:
                st.session_state.conversation.append({
                    "role": "tool_call",
                    "content": tool_call
                })
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Reset processing flag
        st.session_state.processing = False

# User input
if user_input := st.chat_input("Type your message here...", disabled=st.session_state.processing):
    # Display user message
    st.chat_message("user", avatar="./utils/user.png").markdown(user_input)
    
    # Process the message
    process_message(user_input)
