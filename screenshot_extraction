
!sudo apt-get install tesseract-ocr
!pip install pytesseract
!pip install pillow
from PIL import Image
import pytesseract

image_path='/content/sample_data/Screenshot 2024-11-25 163139.png'

image=Image.open(image_path)
text=pytesseract.image_to_string(image)
print(text)
!pip install langchain_mistralai --quiet
!pip install langchain --quiet
!pip install langchain_community --quiet
from google.colab import userdata

api_key=userdata.get('MISTRAL_AI_API_KEY')
from langchain_mistralai import ChatMistralAI

llm=ChatMistralAI(api_key=api_key)
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
template=PromptTemplate(
    input_variables=['context',],
    template="""
    You are an assistant who can extract SQL 
    queries from the input text.
    The query should be executable in 
    an sql database and should not contain '\'
    
    text: {context}
    answer:
    """
)

chain=template | llm

result=chain.invoke(text)
result.content.split(';')[0]
