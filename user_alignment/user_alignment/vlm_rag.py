import time
import chromadb
import ollama
import re
import faster_whisper
import ast
import numpy as np
from user_alignment.utils import get_useful_doc, get_draft


class VLMRAG():
    def __init__(self):

        # Initialize the ChromaDB client
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

        # Set of example documents to be used for rag
        self.documents = []

        # Generate a vision response for the image
        self.prompt_vlm = """
        You're a robot assistant. Please look at the image and describe each object on the table simply. Ignore the table, any robot arms. Only describe the objects that are on the table.
        Identify and list **all** visible objects **on the table**. Return the result as a valid Python list of strings.
        If the table is empty, return None.
        """

    def infer(self, image_path):
        """
        Generate a vision response for the image.
        """
        
        ###
        #Get image description (object list) from the VLM
        response = ollama.generate(
            model='llama3.2-vision',
            prompt=self.prompt_vlm,
            images=[image_path],
            options={
                "temperature": 0.0,
                "num_predict": 1024
            }
        )

        final_res = []
    
        im_desc = response.get("response", "")
        
        if im_desc[:4] == 'None':
            return(None, None)
        
        print(f"Image description: {im_desc}")
        match = re.search(r'\[\s*.*?\s*\]', im_desc, re.DOTALL)
        if match:
            obj_list_str = match.group(0)
            obj_list = ast.literal_eval(obj_list_str)  #
            print(f"Extracted object list: {obj_list}")
            self.obj = obj_list[0]
        else:
            print("List not found")
            
        ###
        #Get the suggested action for the first object in the list (the action will be returned at the end, rendering the for loop useless. It can be changed to a print or other to get all the suggested actions.

        for obj in obj_list:

            context = []
            print('Collection: ', self.collection) #Display available corrections
            docs = get_useful_doc(self.collection,obj)
            print('Docs: ', docs) #Display selected correction
            prompt = get_draft(docs, obj)

            context.append({'role':'user','content':prompt})
            
            start_time = time.time()
            response = ollama.generate(
                model='qwen3:4b',
                prompt=prompt
            )
            
            response = re.sub(r'<think>.*?</think>\s*', '', response.get('response', ''), flags=re.DOTALL)
            print(f"Response from the model: {response}") #Display initial object suggection

            context.append({'role':'assistant','content':response})

            corrections = []

            id_coll = 0
            info = input("Add an information for the model to correct the plan: ") #Accept user feedback
            corrections.append(info)

            context.append({'role':'user','content':info})
          
            if info != 'ok': #If the info is a correction, add it to the database #TODO: embed object id instead to robustify comparison?
                emb = ollama.embed(model="mxbai-embed-large", input=info)
                embeddings = emb["embeddings"]
                self.collection.add(
                    ids=[str(id_coll)],
                    embeddings=embeddings,
                    documents=[info]
                )
                id_coll += 1

            docs = get_useful_doc(self.collection, obj, 0.5)
            print(f"Useful documents for the task: {docs}")
            prompt = f"User feedback: {info}. Please update the proposed action."#get_draft(docs, obj)

            response = ollama.chat('qwen3:4b',context) #TODO: check this, verify context building, add data saving (ideally with tool)

            response = re.sub(r'<think>.*?</think>\s*', '', response.message.content, flags=re.DOTALL)
            print(f"Response from the model after adding: {response}")
            
            return(self.obj, response) #Return object and action
