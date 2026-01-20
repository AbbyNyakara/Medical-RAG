from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv  # api key
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path
import logging
from langchain_core.output_parsers import StrOutputParser

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

class GenerateService:
    '''
    Generate answer based on Open AI model
    '''

    def __init__(self, model: str = 'gpt-4-turbo', timeout: int = 30, retry_attempts=3, temperature: int = 0,  max_tokens: int = 500):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout  # in seconds
        self.retry_attempts = retry_attempts
        self.max_tokens = max_tokens

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            request_timeout=timeout,
            max_retries=retry_attempts
        )

    def generate_with_llm(self, prompt: PromptTemplate, context: str, question: str):
        '''Generate answer from prompt template'''
        formatted_prompt = prompt.format(context=context, question=question)
        if len(formatted_prompt) > 12000:
            logger.warning("Prompt is very long, may exceed token limits")
        response = self.llm.invoke(formatted_prompt)
        return response.content
    
    def generate_with_chain(self, prompt, context,question):
        """
        Use LangChain LCEL chain
        More robust and composable
        """
        chain = prompt | self.llm | StrOutputParser()

        result = chain.invoke({
                "context": context,
                "question": question
            })
        logger.info("Chain generated response")
        return result
    
    

