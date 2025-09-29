import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

class BERTPredictor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def predict(self, questions):
        # Tokenize inputs
        encodings = self.tokenizer(questions, truncation=True, padding=True, max_length=120, return_tensors='pt')
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        
        # Convert logits to predictions (0-5)
        predictions = outputs.logits.argmax(dim=1).cpu().numpy()
        # Convert to original label range (1-6) and add 'K' prefix
        return ['K' + str(pred + 1) for pred in predictions]

    def process_excel(self, input_path, output_path):
        # Read input file
        df = pd.read_excel(input_path)
        
        # Ensure required columns exist
        if 'Question' not in df.columns:
            raise ValueError("Input Excel must contain 'Question' column")
            
        # Rename 'Label' to 'Actual Label' if exists
        if 'Label' in df.columns:
            df = df.rename(columns={'Label': 'Actual Label'})
        elif 'Actual Label' not in df.columns:
            # If neither exists, create empty 'Actual Label' column
            df['Actual Label'] = None
            
        # Get predictions
        questions = df['Question'].astype(str).tolist()
        predictions = self.predict(questions)
        
        # Insert predicted column after 'Actual Label'
        actual_label_idx = df.columns.get_loc('Actual Label')
        df.insert(actual_label_idx + 1, 'Predicted Label', predictions)
        
        # Save to output file
        df.to_excel(output_path, index=False)
        return output_path