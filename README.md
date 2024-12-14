# Vision-Scraper

## Overview
**Vision-Scraper** is an advanced agent designed to perform web scraping and web searching tasks with a focus on leveraging vision-based insights. It provides a framework for building an "agentic AI" capable of dynamically handling complex queries related to scraping and searching.

### Deployed Application
Access the deployed application here: [Vision-Scraper on Hugging Face](https://huggingface.co/spaces/nerdthingz/vision-scraper)

### Key Features
- **Web Scraping**: Extracts specific data from web pages using custom scraping tools.
- **Web Searching**: Retrieves search results using the Tavily API.
- **Vision Integration**: Utilizes vision-based tools to interpret images for web scraping guidance.
- **Dynamic Agent Workflow**: Adapts to user queries by selecting the appropriate tool for the task.
- **Error Handling**: Detects and reports errors during scraping or code execution to ensure robust operations.

## How It Works
1. **Agent Decision-Making**
   - Based on the user's query, the agent determines whether to:
     - Perform a **web search** using Tavily.
     - Execute **web scraping** with a custom scraping tool.

2. **Web Scraping Workflow**
   - The agent receives a basic HTML structure of the target page.
   - It prompts an LLM (Language Model) to generate code for extracting the requested data.
   - The generated code is executed, and the extracted data is returned to the user.

3. **Vision-Based Scraping**
   - If the user provides an image, the image is processed using **Gemini**
   - Gemini provides insights into how the scraping should be performed.
   - The agent uses these insights to guide the scraping process.

4. **Error Handling**
   - The agent identifies and reports any issues encountered during scraping or code execution.
   - This feedback is used to iteratively improve the scraping process.

## Use Case
Vision-Scraper demonstrates how to build a sophisticated agentic AI capable of:
- Automating web scraping tasks with minimal manual intervention.
- Integrating vision-based analysis for enhanced functionality.
- Handling errors gracefully to provide reliable outputs.

Feel free to explore, customize, and enhance this project to suit your needs.


