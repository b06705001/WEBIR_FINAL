from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset

def build_data(sentences, MAX_LEN=50, labels=None):
    # tokenizer -----------------------------------------------------------------------
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids = []
    attention_masks = []
    for sen in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sen,                            # Sentence to encode.
                            add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                            max_length = MAX_LEN,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',          # Return pytorch tensors.
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    print("tokenizer step is finished")

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    if labels is not None:
        labels = torch.tensor(labels)
        # Combine the training inputs into a TensorDataset.
        return TensorDataset(input_ids, attention_masks, labels)
    else:
        return TensorDataset(input_ids, attention_masks)

