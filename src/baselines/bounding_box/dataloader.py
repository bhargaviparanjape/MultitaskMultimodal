from utilities import *

class DataLoader(object):
    def __init__(self,args):
        self.args = args
        #self.vocabulary = Vocabulary(args.dictionary_path)
        self.vocabulary = Vocabulary()
        self.bigrams = defaultdict(lambda : 0)
        self.unigrams = defaultdict(lambda : 0)
        self.trigrams = defaultdict(lambda : 0)
        self.fourgrams = defaultdict(lambda: 0)

        self.quadcount = 0
        self.tricount = 0
        self.bicount = 0
        self.unicount = 0

    def visualize(self, data):
        print(len(data))
        data = data[:30]
        for sent in data:
            print(" ".join([self.vocabulary.id2word[token] for token in sent]))

    def get_pretrained_emb(self, word_to_id, dim):
        word_emb = []

        for _ in range(len(word_to_id)):
            word_emb.append(np.random.uniform(-math.sqrt(3.0 / dim), math.sqrt(3.0 / dim), size=dim))

        print("length of dict: ".format( len(word_to_id)))
        pretrain_word_emb = {}
        if self.args.pretrain_path is not None and os.path.exists(self.args.pretrain_path):
            print("Loading pretrained embeddings from %s." % (self.args.pretrain_path))
            for line in codecs.open(self.args.pretrain_path, "r", "utf-8", errors='replace'):
                items = line.strip().split()
                if len(items) == dim + 1:
                    try:
                        pretrain_word_emb[items[0]] = np.asarray(items[1:]).astype(np.float32)
                    except ValueError:
                        continue

            not_covered = 0
            for word, id in word_to_id.iteritems():
                if word in pretrain_word_emb:
                    word_emb[id] = pretrain_word_emb[word]
                elif word.lower() in pretrain_word_emb:
                    word_emb[id] = pretrain_word_emb[word.lower()]
                else:
                    not_covered += 1
            print("Word number not covered in pretrain embedding: ".format(not_covered))

        word_emb[0] = np.zeros(dim)
        emb = np.array(word_emb, dtype=np.float32)
        return emb

    def load_test_dataset(self, question_path,analysze=False):
        questions = json.load(open(question_path))['questions']
        sorted_questions = sorted(questions, key=lambda x: x['question_id'])
        metainfo = []
        datapoints = []  # currently concatenating only answers
        count = 0
        for question in sorted_questions:
            image_id = question['image_id']

            datapoints.append(self.preprocess_tokens(question['question'], init=False,analysze=analysze))
            count += 1

            if count % 100000 == 0:
                print("Processed {0} points".format(count))

        return metainfo, datapoints

    def load_dataset(self, question_path,analysze= False):
        questions = json.load(open(question_path))['questions']
        sorted_questions = sorted(questions, key=lambda x :x['question_id'])
        metainfo = []
        datapoints = []  # currently concatenating only answers
        count = 0

        for question in sorted_questions:
            image_id = question['image_id']
            datapoints.append(image_id)
            count +=1

            if count % 100000 == 0:
                print("Processed {0} points".format(count))

        return datapoints

    def load_captions(self, path, analysze= False, init=False):
        annotations = json.load(open(path))["annotations"]
        datapoints = []  # currently concatenating only answers
        count = 0
        for annot in annotations:
	    image_id = annot["image_id"]
	    datapoints.append(image_id)
            count += 1

            if count % 100000 == 0:
                print("Processed {0} points".format(count))
	return datapoints

    def debug(self, sentences):
        tokens = []
        for sentence in sentences:
            tokens.append(self.preprocess_tokens(sentence,init=False))


        return tokens

    def analysis(self):
        fout = codecs.open("./error_analysis.txt","w")
        quadgrams = sorted(self.fourgrams.iteritems(),key=lambda (k,v):v, reverse=True)[:20]
        fout.write("QUADGRAMS: NUM: "  + str(self.quadcount) + "\n")
        for quad in quadgrams:
            fout.write(quad[0] + "@@"  + str((self.fourgrams[quad[0]])) + "\n")
        fout.write("\n")

        trigrams = sorted(self.trigrams.iteritems(),key=lambda (k,v):v, reverse=True)[:20]
        fout.write("TRIGRAMS: NUM: " + str(self.tricount) + "\n")
        for tri in trigrams:
            fout.write(tri[0] +  "@@"  + str(self.trigrams[tri[0]])+ "\n")
        fout.write("\n")

        bigrams = sorted(self.bigrams.iteritems(),key=lambda (k,v):v, reverse=True)[:20]
        fout.write("BIGRAMS: NUM: " + str(self.bicount) + "\n")
        for x in bigrams:
            fout.write(x[0] +   "@@"  + str(self.bigrams[x[0]]) + "\n")
        fout.write("\n")

        unigrams = sorted(self.unigrams.iteritems(),key=lambda (k,v):v,reverse=True)[:20]
        fout.write("UNIRAMS: NUM: " + str(self.unicount) + "\n")
        for x in unigrams:
            fout.write(x[0] +  "@@"  + str(self.unigrams[x[0]])   + "\n")
        fout.write("\n")

    def preprocess_tokens(self, text, init, analysze = False):
        tokens = text.lower().replace(",","").replace('?',' <eos>').replace("\'s",' \'s').split()
        self.getNgrams(analysze, tokens)

        token_ids = []
        for token in tokens:
            token_ids.append(self.vocabulary.add_and_get_index(token,init))

        return token_ids

    def getNgrams(self, analysze, tokens):
        if analysze:
            all_ngrams = []
            for i in range(4, 0, -1):
                all_ngrams += zip(*[tokens[j:] for j in range(i)])
            for ngram in all_ngrams:
                ngram = list(ngram)
                if len(ngram) == 4:
                    self.fourgrams[" ".join(ngram)] += 1
                    self.quadcount += 1
                elif len(ngram) == 3:
                    self.trigrams[" ".join(ngram)] += 1
                    self.tricount += 1
                elif len(ngram) == 2:
                    self.bigrams[" ".join(ngram)] += 1
                    self.bicount += 1
                elif len(ngram) == 1:
                    self.unigrams[" ".join(ngram)] += 1
                    self.unicount += 1


class Vocabulary(object):
    def __init__(self, path=None ):
        if path:
            self.word2id, self.id2word = pickle.load(open(path, 'rb'))
            print("Loading vocabulary of length {0}".format(len(self.word2id)))

        else:
            self.word2id = {}
            self.id2word = []

        if 'unk' not in self.word2id:
            self.add_and_get_index('unk',init=True)
        if '<eos>' not in self.word2id:
            self.add_and_get_index('<eos>',init=True)
        if '<pad>' not in self.word2id:
            self.add_and_get_index('<pad>',init=True)

    def add_and_get_index(self, word, init=False):
        if word in self.word2id:
            return self.word2id[word]
        else:
            if init:
                length = len(self.word2id)
                self.word2id[word] = length
                self.id2word.append(word)
                return length
            else:
                return self.word2id['unk']

    def add_and_get_indices(self, words):
        return [self.add_and_get_index(word) for word in words]

    def get_index(self, word):
        return self.word2id.get(word, self.word2id['unk'])

    def get_length(self):
        return len(self.word2id)

    def get_word(self,index):
        if index < len(self.id2word):
            return self.id2word[index]
        else:
            return ""
