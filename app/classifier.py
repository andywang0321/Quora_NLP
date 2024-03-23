import streamlit as st
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def load_model():
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    PATH = 'finetuned_bert_classifier'
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model

model = load_model()

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case = True
)

def preprocessing(input_text, tokenizer):
    '''
    Returns <class transformers.tokenization_utils_base.BatchEncoding>
    with the following fields:
        - input_ids:        list of token ids
        - token_type_ids:   list of token type ids
        - attention_mask:   list of indices (0, 1) specifying which tokens
                            should be considered by the model 
                            (return_attention_mask = True)
    '''
    return tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=32,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

def predict(new_sentence):
    # We need Token IDs and Attention Mask for inference on the new sentence
    test_ids = []
    test_attention_mask = []

    # Apply the tokenizer
    encoding = preprocessing(new_sentence, tokenizer)

    # Extract IDs and Attention Mask
    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)

    # Forward pass, calculate logit predictions
    with torch.no_grad():
        output = model(test_ids, token_type_ids = None, attention_mask = test_attention_mask)

    insincere = True if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else False

    return insincere

def show_classifier_page():
    st.title('Is my question appropriate? 👀')

    placeholder_question = 'Why can some birds fly, but others can\'t?'

    go = st.text_input('Your question here 👇',
                  key='user_question',
                  placeholder=placeholder_question
    )

    #go = st.button('Go!')

    if go:
        insincere = predict(st.session_state.user_question)

        if insincere:
            st.error('Shhh! Are you trying to get cancelled on the Internet? :face_with_symbols_on_mouth:')
        else:
            st.success('Great question! I wonder that too... :thinking_face:')


    