import re
import pathlib
import ollama

class LLMClient:
    """LLM Client class. Manages the setup of the LLM and pure generation of plan framework."""
    
    def __init__(self, model_name="llama3.2", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self._load_model()

    def _load_model(self):
        try :
            client = ollama.Client(host=self.host)
            models = client.list()
            model_pulled = any(model.model == self.model_name for model in models.models)
            
            if not model_pulled :
                # print(f"Model {self.model_name} not found at {self.host}. Pulling it form Ollama...")
                client.pull(self.model_name)

            print(f"Successfully loaded model: {self.model_name}")
            self.client = client
            
        except Exception as e :
            print(f"Error loading model {self.model_name} from {self.host}")
            raise
    
    def load_prompt_template(self, prompt_file):
        with open(prompt_file) as f :
            return f.read()

    def generate(self, prompt):
        """
        Generates the answer from prompt.
        """
        # TODO check if you can format the answer 

        response = self.client.generate(
            model=self.model_name,
            prompt=prompt
        )

        return response['response']

if __name__=="__main__":
    dir = pathlib.Path(__file__).parent.resolve()
    print(dir)

    llm = LLMClient()
    print(llm._generate("why is the sky blue ?"))