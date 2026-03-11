import os
import sys
import argparse
from agent import initialize_agent

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='input a question for the agent to analyze')

    return parser

def main():

    parser = argparse.ArgumentParser(description='agent interface')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()    
    query = args.input

    agent = initialize_agent()

    for chunk in agent.stream({
        "messages": [{"role": "user", "content": query}]
    }, stream_mode="values"):
        # Each chunk contains the full state at that point
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            print(f"Agent: {latest_message.content}")
        elif latest_message.tool_calls:
            print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")    
    
if __name__ == '__main__':
    main()
