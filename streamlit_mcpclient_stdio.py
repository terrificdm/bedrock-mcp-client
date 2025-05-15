import streamlit as st
import boto3
import json
import threading
import asyncio
from typing import List, Dict, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import atexit

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

# Define Message utility class
class Message:
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
                        "required": tool["input_schema"].get("required", [])
                    }
                }
            }
        } for tool in tools_list]

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

# Initialize session state
default_session_state = {
    'conversation': [],
    'connected': False,
    'bedrock_client': None,
    'mcp_servers': {},
    'tools': [],
    'tool_server_map': {},
    'tool_use_history': {},
    'current_config_path': "",
    'current_region': "",
    'current_model_id': DEFAULT_MODEL
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
        col1, col2 = st.columns([1,3.5])
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

# Async function definitions
async def mcp_server_request(command, args, action_type, **kwargs):
    """Generic async function to handle MCP server requests"""
    server_params = StdioServerParameters(command=command, args=args)
    
    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()
            
            if action_type == "list_tools":
                response = await session.list_tools()
                tools = []
                for tool in response.tools:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
                return tools
            
            elif action_type == "call_tool":
                tool_name = kwargs.get("tool_name")
                tool_args = kwargs.get("tool_args")
                result = await session.call_tool(tool_name, tool_args)
                return {"text": result.content[0].text}

async def list_tools_async(command, args):
    """Asynchronously get tool list"""
    return await mcp_server_request(command, args, "list_tools")

async def call_tool_async(command, args, tool_name, tool_args):
    """Asynchronously call a tool"""
    return await mcp_server_request(command, args, "call_tool", 
                                   tool_name=tool_name, tool_args=tool_args)

# MCP Server Manager class
class MCPServerManager:
    """Class to manage MCP server operations"""
    
    @staticmethod
    def start_servers(config_path):
        """Start MCP servers from configuration file"""
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if "mcpServers" not in config or not config["mcpServers"]:
                st.error("No servers defined in configuration file")
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
            st.error(f"Error loading server configuration: {str(e)}")
            return False
    
    @staticmethod
    def get_tools():
        """Get tools from all MCP servers"""
        tools = []
        tool_server_map = {}
        server_tools_map = {}  # Map to store tools by server
        
        for server_name, server_config in st.session_state.mcp_servers.items():
            try:
                # Use event loop manager to run async function
                server_tools = st.session_state.asyncio_loop.run_coroutine(
                    list_tools_async(
                        server_config["command"], 
                        server_config["args"]
                    )
                )
                
                # Store tools for this server
                server_tools_map[server_name] = [tool["name"] for tool in server_tools]
                
                # Add server name to tools
                for tool in server_tools:
                    tool["server_name"] = server_name
                    tool_server_map[tool["name"]] = server_name
                
                tools.extend(server_tools)
                
            except Exception as e:
                st.error(f"Error getting tools from server {server_name}: {str(e)}")
        
        return tools, tool_server_map, server_tools_map
    
    @staticmethod
    def call_tool(tool_name, tool_args, server_name):
        """Call a tool on a specific server"""
        try:
            server_config = st.session_state.mcp_servers[server_name]
            
            # Use event loop manager to run async function
            result = st.session_state.asyncio_loop.run_coroutine(
                call_tool_async(
                    server_config["command"],
                    server_config["args"],
                    tool_name,
                    tool_args
                )
            )
            
            return result
            
        except KeyError:
            error_msg = f"Server '{server_name}' not found in configuration"
            st.error(error_msg)
            return {"text": error_msg}
        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {str(e)}"
            st.error(error_msg)
            return {"text": error_msg}

# Define a function to initialize the Bedrock client
def initialize_bedrock_client(region):
    with st.spinner(f"Initializing Bedrock client in region: {region}"):
        st.session_state.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region
        )
        return True

# Auto-connect to MCP servers on app startup or refresh
if not st.session_state.connected:
    with st.spinner("Connecting to MCP servers..."):
        # Initialize Bedrock client
        initialize_bedrock_client(region_name)
        
        # Start MCP servers
        if MCPServerManager.start_servers(config_path):
            # Get tools from servers
            tools, tool_server_map, server_tools_map = MCPServerManager.get_tools()
            
            if tools:
                st.session_state.tools = tools
                st.session_state.tool_server_map = tool_server_map
                st.session_state.server_tools_map = server_tools_map
                st.session_state.connected = True
                st.success(f"Connected to {len(st.session_state.mcp_servers)} MCP servers with {len(tools)} tools, check left pannel for details!")
            else:
                st.error("No tools found on any server")
        else:
            st.error("Failed to start MCP servers")

# Define a function to reconnect to MCP servers
def reconnect_mcp_servers(reason=""):
    with st.spinner(f"Reconnecting... {reason}"):
        # Reset connection status
        st.session_state.connected = False
        st.session_state.mcp_servers = {}
        st.session_state.tools = []
        st.session_state.tool_server_map = {}
        
        # Initialize Bedrock client
        initialize_bedrock_client(region_name)
        
        # Start MCP servers
        if MCPServerManager.start_servers(config_path):
            # Get tools from servers
            tools, tool_server_map, server_tools_map = MCPServerManager.get_tools()
            
            if tools:
                st.session_state.tools = tools
                st.session_state.tool_server_map = tool_server_map
                st.session_state.server_tools_map = server_tools_map
                st.session_state.connected = True
                return True
            else:
                st.error("No tools found on any server")
                return False
        else:
            st.error("Failed to start MCP servers")
            return False

# If configuration path changes, trigger reconnection
if st.session_state.current_config_path != config_path and st.session_state.current_config_path != "":
    reconnect_mcp_servers("Configuration file changed")
    
# If region changes, update Bedrock client
if region_changed:
    initialize_bedrock_client(region_name)
    st.success(f"Switched to region: {region_name}")

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
    st.session_state.conversation = []
    # Clear tool call history
    if 'tool_use_history' in st.session_state:
        st.session_state.tool_use_history = {}

with st.chat_message("assistant", avatar="./utils/assistant.png"):
        st.write("I am an AI chatbot powered by Amazon Bedrock, what can I do for you？💬")

# Display conversation history
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.chat_message("user", avatar="./utils/user.png").write(message["content"][0]["text"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant", avatar="./utils/assistant.png"):
            for content_item in message["content"]:
                if "text" in content_item:
                    st.write(content_item["text"])
    elif message["role"] == "tool_call":
        st.info(message["content"])

# Function to handle tool use
def handle_tool_use(message, messages, tools):
    for content_item in message['content']:
        if 'toolUse' in content_item:
            tool_info = content_item['toolUse']
            
            # Call the tool
            tool_name = tool_info['name']
            tool_args = tool_info['input']
            tool_use_id = tool_info['toolUseId']
            
            # Record tool call
            st.session_state.tool_use_history[tool_use_id] = {
                "tool_name": tool_name,
                "tool_args": tool_args,
                "has_result": False
            }
            
            # Find which server has this tool
            server_name = st.session_state.tool_server_map.get(tool_name)
            if not server_name:
                st.error(f"Tool '{tool_name}' is not available")
                return
            
            # Display tool call info and add to conversation history
            tool_call_message = f"Calling tool {tool_name} on server {server_name}"
            with st.spinner(f"Executing tool: {tool_name}..."):
                st.info(tool_call_message)
                
                # Add tool call to conversation for display
                st.session_state.conversation.append({
                    "role": "tool_call",
                    "content": tool_call_message
                })
                
                # Call the tool using our synchronous helper
                result = MCPServerManager.call_tool(tool_name, tool_args, server_name)
            
            # Add the tool result to conversation for Bedrock (but not for display)
            
            # Mark tool call as having a result
            st.session_state.tool_use_history[tool_use_id]["has_result"] = True
            
            # Create a new conversation history for the continuation
            continuation_messages = []
            
            # Only include messages with matching tool call results, or messages without tool calls
            for msg in messages:
                if msg["role"] == "assistant":
                    # Check if this message contains tool calls
                    has_unmatched_tool_use = False
                    for item in msg.get("content", []):
                        if "toolUse" in item:
                            tool_id = item["toolUse"]["toolUseId"]
                            if tool_id not in st.session_state.tool_use_history or not st.session_state.tool_use_history[tool_id]["has_result"]:
                                has_unmatched_tool_use = True
                                break
                    
                    if not has_unmatched_tool_use:
                        continuation_messages.append(msg)
                else:
                    continuation_messages.append(msg)
            
            # Add the current assistant message
            continuation_messages.append(message)
            
            # Add the tool result
            tool_result_message = {
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"json": {"text": result["text"]}}]
                    }
                }]
            }
            continuation_messages.append(tool_result_message)
            
            # Continue the conversation with the tool result
            continue_conversation(continuation_messages, tools)

# Function to process streaming response from Bedrock
def process_streaming_response(response, placeholder=None):
    message = {"content": []}
    text = ''
    tool_use = {}
    stop_reason = ""
    
    # Stream the response
    for chunk in response['stream']:
        chunk_type = next(iter(chunk))  # Get the first key in the chunk
        chunk_data = chunk[chunk_type]
        
        if chunk_type == 'messageStart':
            message['role'] = chunk_data['role']
        
        elif chunk_type == 'contentBlockStart':
            if 'toolUse' in chunk_data.get('start', {}):
                tool = chunk_data['start']['toolUse']
                tool_use = {'toolUseId': tool['toolUseId'], 'name': tool['name']}
        
        elif chunk_type == 'contentBlockDelta':
            delta = chunk_data['delta']
            if 'toolUse' in delta:
                if 'input' not in tool_use:
                    tool_use['input'] = ''
                tool_use['input'] += delta['toolUse']['input']
            elif 'text' in delta:
                text += delta['text']
                # Update the message placeholder if provided
                if placeholder:
                    placeholder.markdown(text)
        
        elif chunk_type == 'contentBlockStop':
            if 'input' in tool_use:
                try:
                    tool_use['input'] = json.loads(tool_use['input'])
                except json.JSONDecodeError:
                    tool_use['input'] = {"text": tool_use['input']}
                message['content'].append({'toolUse': tool_use})
                tool_use = {}
            else:
                message['content'].append({'text': text})
                text = ''
        
        elif chunk_type == 'messageStop':
            stop_reason = chunk_data['stopReason']
    
    # Handle any remaining text
    if text and not any(item.get('text') == text for item in message['content']):
        message['content'].append({'text': text})
    
    return message, stop_reason

# Function to continue conversation after tool call
def continue_conversation(messages, tools):
    try:
        # Make request to Bedrock
        response = st.session_state.bedrock_client.converse_stream(
            modelId=model_id,
            messages=messages,
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature
            },
            toolConfig={"tools": tools}
        )
        
        # Create a new chat message for the continued response
        with st.chat_message("assistant", avatar="./utils/assistant.png"):
            continue_placeholder = st.empty()
            
            # Process the streaming response
            message, stop_reason = process_streaming_response(response, continue_placeholder)
        
        # Add the assistant message to conversation
        st.session_state.conversation.append(message)
        
        # Handle nested tool calls if needed
        if stop_reason == "tool_use":
            handle_tool_use(message, messages, tools)
                
    except Exception as e:
        st.error(f"Error in continuation: {str(e)}")
        import traceback
        traceback.print_exc()

# Process a message with tool handling
def process_message(user_input):
    if not st.session_state.connected:
        st.error("Please connect to MCP servers first")
        return
    
    with st.spinner("Processing your request..."):
        # Add user message to conversation
        user_message = {"role": "user", "content": [{"text": user_input}]}
        st.session_state.conversation.append(user_message)
    
    # Prepare conversation history for Bedrock - only include user and assistant messages
    # But ensure tool calls and tool results match
    bedrock_messages = []
    tool_use_ids = set()  # Track tool call IDs we've seen
    tool_result_ids = set()  # Track tool result IDs we've seen
    
    # First collect all tool call and result IDs
    for msg in st.session_state.conversation:
        if msg["role"] == "assistant":
            for content_item in msg.get("content", []):
                if "toolUse" in content_item:
                    tool_use_ids.add(content_item["toolUse"]["toolUseId"])
        elif msg["role"] == "user":
            for content_item in msg.get("content", []):
                if "toolResult" in content_item:
                    tool_result_ids.add(content_item["toolResult"]["toolUseId"])
    
    # Only include messages with matching tool call results, or messages without tool calls
    for msg in st.session_state.conversation:
        if msg["role"] == "assistant":
            # Check if this message contains tool calls
            has_tool_use = False
            tool_use_in_msg = []
            for content_item in msg.get("content", []):
                if "toolUse" in content_item:
                    has_tool_use = True
                    tool_use_id = content_item["toolUse"]["toolUseId"]
                    if tool_use_id in tool_result_ids:
                        tool_use_in_msg.append(tool_use_id)
            
            # If message contains tool calls but no matching results, skip this message
            if has_tool_use and not tool_use_in_msg:
                continue
                
            # Otherwise add to Bedrock messages
            bedrock_messages.append(msg)
        elif msg["role"] == "user":
            # Check if this message contains tool results
            has_tool_result = False
            for content_item in msg.get("content", []):
                if "toolResult" in content_item:
                    has_tool_result = True
                    tool_use_id = content_item["toolResult"]["toolUseId"]
                    # Only add tool results with matching tool calls
                    if tool_use_id not in tool_use_ids:
                        continue
            
            # Add to Bedrock messages
            bedrock_messages.append(msg)
    
    # Prepare tools for Bedrock
    bedrock_tools = Message.to_bedrock_format([{k: v for k, v in tool.items() if k != 'server_name'} 
                                             for tool in st.session_state.tools])
    
    try:
        # Create a placeholder for the assistant's response
        with st.chat_message("assistant", avatar="./utils/assistant.png"):
            message_placeholder = st.empty()
            
            # Make request to Bedrock
            response = st.session_state.bedrock_client.converse_stream(
                modelId=model_id,
                messages=bedrock_messages,
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature
                },
                toolConfig={"tools": bedrock_tools}
            )
            
            # Process the streaming response
            message, stop_reason = process_streaming_response(response, message_placeholder)
        
        # Add the assistant message to conversation
        st.session_state.conversation.append(message)
        
        # Handle tool use if needed
        if stop_reason == "tool_use":
            handle_tool_use(message, bedrock_messages, bedrock_tools)
                    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

# User input
if user_input := st.chat_input("Type your message here..."):
    # Display user message
    st.chat_message("user", avatar="./utils/user.png").write(user_input)
    
    # Process the message
    process_message(user_input)
