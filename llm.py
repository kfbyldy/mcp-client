import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

from openai import AsyncOpenAI

load_dotenv()  # load environment variables from .env  


class MCPClient:
    def __init__(self):
        # Initialize session and client objects  
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
        )

        # methods will go here
        #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   deepseek/deepseek-r1:free
        #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   deepseek/deepseek-chat-v3-0324:free
        #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 google/gemini-2.5-pro-preview-03-25   Insufficient credits
        #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 microsoft/mai-ds-r1:free  No endpoints found that support tool use.
        #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 qwen/qwen2.5-coder-7b-instruct
        #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 meta-llama/llama-3.1-8b-instruct

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server  

        Args:            server_script_path: Path to the server script (.py or .js)        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools  
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]


        # 初始化 LLM API 调用
        response = await self.client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=messages
        )

        final_text = []
        message = response.choices[0].message
        final_text.append(message.content or "")



        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        # await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())