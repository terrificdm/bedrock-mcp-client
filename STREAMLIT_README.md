# Streamlit MCP Client for AWS Bedrock

### Features
* Interactive web UI for conversations with AWS Bedrock models
* Real-time streaming responses
* Support for multiple MCP tool servers
* Multi-turn conversations with context retention

### Prerequisites
* Python 3.10 or higher
* Configured AWS credentials with Bedrock access permissions
* Installed UV package manager

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd bedrock-mcp-client
   ```

2. Create and activate a virtual environment:
   ```
   uv venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   uv pip install -r requirements.txt
   ```

### Usage

1. Start the Streamlit application:
   ```
   streamlit run streamlit_mcpclient_stdio.py
   ```

2. Configure the application in the sidebar:
   - Enter the path to your MCP configuration file (default: `mcp_config.json`)
   - Select the AWS Bedrock model to use
   - Choose the AWS region
   - Adjust model parameters as needed

3. Start chatting with the model in the main interface


### Note

* This application supports stdio (Local) MCP servers only
* All code is developed with [Amazon Q Developer](https://aws.amazon.com/q/developer/build/)