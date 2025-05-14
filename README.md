# MCP client for AWS BedRock  

### Features:  
* Supports multi-turn conversations with streaming output     
* Supports connecting to multiple MCP servers through modifying `mcp_config.json` file  
* Supports multiple parameters,such as `-c` `-r` `-m` etc. Type `-h` for more help information  
* Supports stdio(Local) MCP server only, stay tuned...

### Prerequisites:  
* Python 3.10 or higher  
* Configured AWS credentials for accessing BedRock  
* Installed UV package manager  

### How to use:
1. `git clone the repo`  
2. `cd bedrock-mcp-client  `
3. `uv venv`  
4. `source .venv/bin/activate ` 
5. `uv run mcp_client_stdio.py`  

### Note:
* All of codes are developed by [Amazon Q developer](https://aws.amazon.com/q/developer/build/?trk=e7e88218-8649-422c-8f4c-2af954cf1e1a&sc_channel=ps&ef_id=CjwKCAjwz_bABhAGEiwAm-P8YavEE_r3ZpddJGDH4jIBBW4qfzFheIUlq70fskbqh4uUx7-mmdyMahoCx7QQAvD_BwE:G:s&s_kwcid=AL!4422!3!698165432143!e!!g!!amazon%20q%20developer!21048268275!168533076464&gad_campaignid=21048268275&gbraid=0AAAAADjHtp9gARQnJTW0BUqu5Vq1CU6hI&gclid=CjwKCAjwz_bABhAGEiwAm-P8YavEE_r3ZpddJGDH4jIBBW4qfzFheIUlq70fskbqh4uUx7-mmdyMahoCx7QQAvD_BwE) based on the reference [Model Context Protocol (MCP) and Amazon Bedrock](https://community.aws/content/2uFvyCPQt7KcMxD9ldsJyjZM1Wp/model-context-protocol-mcp-and-amazon-bedrock?lang=en)    
* type `uv run mcp_client_stdio.py -h` for more useages



