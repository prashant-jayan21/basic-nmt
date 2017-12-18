import numpy as np
import tensorflow as tf
import operator

def loadVocabAndEmbeddings(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.rstrip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded vocab and embeddings!')
    file.close()
    embeddings = np.asarray(embd, dtype=np.float32)
    return vocab, embeddings

def loadVocabAndEmbeddings_targetLang(filename, train_sentences, vocab_size, unk_token):

    # get vocab dict
    vocab_dict = {}
    for sentence in train_sentences:
        tokens = sentence.lower().split(" ")
        for token in tokens:
            if token in vocab_dict:
                vocab_dict[token] += 1
            else:
                vocab_dict[token] = 1

    # sort
    sorted_vocab_dict = sorted(vocab_dict.items(), key=operator.itemgetter(1), reverse=True)

    # get top k tokens
    sorted_vocab_dict_top_k = sorted_vocab_dict[:vocab_size]
    vocab_final = list(map(lambda x: x[0], sorted_vocab_dict_top_k))

    # add unk token
    vocab_final.append(unk_token)

    # load pre-trained vocab and embeddings
    vocab_e = []
    embd_e = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.rstrip().split(' ')
        vocab_e.append(row[0])
        embd_e.append(row[1:])
    print('Loaded vocab and embeddings!')
    file.close()

    # construct final reduced embedding matrix
    embd_final = []
    for token in vocab_final:
        # find index in vocab_e
        token_index = index_without_exception(vocab_e, token)
        if token_index == -1: # get index for UNK token
            print("haha")
            token_index = vocab_e.index(unk_token)

        # get embedding based on index
        token_embedding = embd_e[token_index]

        # insert into embeddings_final
        embd_final.append(token_embedding)

    embeddings_final = np.asarray(embd_final, dtype=np.float32)

    return vocab_final, embeddings_final

def index_without_exception(alist, elem):
    try:
        return alist.index(elem)
    except:
        return -1

def sentenceToTokensIndexed(sentence, vocab, unk_token):
    # TODO: lowercase vocab too?
    tokens = sentence.lower().split(" ") # lower case for case agnostic vocab search and tokenize
    tokens_indexed = []
    for token in tokens:
        token_index = index_without_exception(vocab, token) # find index
        if token_index == -1: # get index for UNK token
            token_index = vocab.index(unk_token)
        tokens_indexed.append(token_index)
    return tokens_indexed

def pad(indexed_tokens_lists, padding_token): # i/p: list of lists
    max_sentence_length = max([len(x) for x in indexed_tokens_lists])
    # pad
    indexed_tokens_lists_padded = []
    for x in indexed_tokens_lists:
        padding_list = [padding_token] * (max_sentence_length - len(x))
        indexed_tokens_lists_padded.append(x + padding_list)
    return max_sentence_length, indexed_tokens_lists_padded

def get_sequence_lengths(indexed_tokens_lists):
    return [len(x) for x in indexed_tokens_lists]

def get_target_weights(decoder_targets, padding_token):
    def f(t, padding_token = padding_token):
        if t == padding_token:
            return 0.0
        else:
            return 1.0
    f_vec = np.vectorize(f)
    return f_vec(decoder_targets)

# append vocab with tgt_sos_id and tgt_eos_id
def append_embedding(embeddings):

    num_rows = embeddings.shape[0]
    tgt_sos_id = num_rows
    tgt_eos_id = num_rows + 1

    num_columns = embeddings.shape[1]
    tgt_sos_embedding = [0] * num_columns # TODO
    tgt_eos_embedding = [1] * num_columns # TODO

    return np.append(embeddings, [tgt_sos_embedding, tgt_eos_embedding], axis = 0), tgt_sos_id, tgt_eos_id

def append_with_sos(indexed_tokens_list, tgt_sos_id):
    return [tgt_sos_id] + indexed_tokens_list

def append_with_eos(indexed_tokens_list, tgt_eos_id):
    return indexed_tokens_list + [tgt_eos_id]

# def loadParallelCorpus(filename_en, filename_hi, num_sentences):
#     with open(filename_en,'r') as myfile:
#         sentences_en = [next(myfile).rstrip() for x in range(num_sentences)]
#     with open(filename_hi,'r') as myfile:
#         sentences_hi = [next(myfile).rstrip() for x in range(num_sentences)]
#     return sentences_en, sentences_hi

def loadParallelCorpus(filename_en, filename_hi, num_sentences, min_length_en, max_length_en):
    file_en = open(filename_en,'r')
    file_hi = open(filename_hi,'r')

    sentences_en = []
    sentences_hi = []

    while len(sentences_en) != num_sentences:
        sentence_en = next(file_en).rstrip()
        tokens_en = sentence_en.split(" ")
        sentence_hi = next(file_hi).rstrip()
        if len(tokens_en) >= min_length_en and len(tokens_en) <= max_length_en:
            sentences_en.append(sentence_en)
            sentences_hi.append(sentence_hi)

    file_en.close()
    file_hi.close()

    return sentences_en, sentences_hi

def get_indexed_tokens_lists(sentences, vocab, unk_token):
    indexed_tokens_lists = []
    for sentence in sentences:
        tokens_indexed = sentenceToTokensIndexed(sentence, vocab, unk_token)
        indexed_tokens_lists.append(tokens_indexed)
    return indexed_tokens_lists

def get_padded_lists(indexed_tokens_lists, padding_token):
    sequence_lengths = get_sequence_lengths(indexed_tokens_lists)
    max_sentence_length, indexed_tokens_lists_padded = pad(indexed_tokens_lists, padding_token)
    return indexed_tokens_lists_padded, sequence_lengths

# For target language ONLY
def append_with_sos_lists(indexed_tokens_lists, tgt_sos_id):
    indexed_tokens_lists_appended_sos = []
    for alist in indexed_tokens_lists:
        appendedlist = append_with_sos(alist, tgt_sos_id)
        indexed_tokens_lists_appended_sos.append(appendedlist)
    return indexed_tokens_lists_appended_sos

def append_with_eos_lists(indexed_tokens_lists, tgt_eos_id):
    indexed_tokens_lists_appended_eos = []
    for alist in indexed_tokens_lists:
        appendedlist = append_with_eos(alist, tgt_eos_id)
        indexed_tokens_lists_appended_eos.append(appendedlist)

    return indexed_tokens_lists_appended_eos

def split_into_batches(sentences_en, sentences_hi, batch_size):
    batched_sentences_en = [sentences_en[x:x+batch_size] for x in range(0, len(sentences_en), batch_size)]
    batched_sentences_hi = [sentences_hi[x:x+batch_size] for x in range(0, len(sentences_hi), batch_size)]

    return list(zip(batched_sentences_en, batched_sentences_hi))

def batch_to_feed_dict(batch, vocab_en, vocab_hi, padding_token, tgt_sos_id, tgt_eos_id, unk_token_en, unk_token_hi):
    sentences_en = batch[0]
    sentences_hi = batch[1]

    indexed_tokens_lists_en = get_indexed_tokens_lists(sentences_en, vocab_en, unk_token_en)
    indexed_tokens_lists_hi = get_indexed_tokens_lists(sentences_hi, vocab_hi, unk_token_hi)

    indexed_tokens_lists_padded_en, sequence_lengths_en = get_padded_lists(indexed_tokens_lists_en, padding_token)

    indexed_tokens_lists_appended_sos_hi = append_with_sos_lists(indexed_tokens_lists_hi, tgt_sos_id)
    indexed_tokens_lists_appended_eos_hi = append_with_eos_lists(indexed_tokens_lists_hi, tgt_eos_id)

    indexed_tokens_lists_appended_sos_padded_hi, sequence_lengths_hi_with_sos = get_padded_lists(indexed_tokens_lists_appended_sos_hi, padding_token)
    indexed_tokens_lists_appended_eos_padded_hi, sequence_lengths_hi_with_eos = get_padded_lists(indexed_tokens_lists_appended_eos_hi, padding_token)

    assert sequence_lengths_hi_with_sos == sequence_lengths_hi_with_eos

    target_weights = get_target_weights(indexed_tokens_lists_appended_eos_padded_hi, padding_token)

    output_dict = {
        "encoder_inputs": indexed_tokens_lists_padded_en,
        "source_sequence_length": sequence_lengths_en,
        "decoder_inputs": indexed_tokens_lists_appended_sos_padded_hi,
        "target_sequence_length": sequence_lengths_hi_with_sos,
        "decoder_targets": indexed_tokens_lists_appended_eos_padded_hi,
        "target_weights": target_weights
    }

    return output_dict
