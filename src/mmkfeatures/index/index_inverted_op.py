import pickle
import unicodedata
from functools import reduce

# List Of English Stop Words
# http://armandbrahaj.blog.al/2009/04/14/list-of-english-stop-words/
_WORD_MIN_LENGTH = 3
_STOP_WORDS = frozenset([
'a', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again',
'against', 'all', 'almost', 'alone', 'along', 'already', 'also','although',
'always','am','among', 'amongst', 'amoungst', 'amount',  'an', 'and', 'another',
'any','anyhow','anyone','anything','anyway', 'anywhere', 'are', 'around', 'as',
'at', 'back','be','became', 'because','become','becomes', 'becoming', 'been',
'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
'between', 'beyond', 'bill', 'both', 'bottom','but', 'by', 'call', 'can',
'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe',
'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight',
'either', 'eleven','else', 'elsewhere', 'empty', 'enough', 'etc', 'even',
'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few',
'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former',
'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get',
'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here',
'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him',
'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc',
'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last',
'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me',
'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly',
'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never',
'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not',
'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only',
'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out',
'over', 'own','part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same',
'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she',
'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some',
'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere',
'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their',
'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby',
'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third',
'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus',
'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two',
'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well',
'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter',
'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which',
'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will',
'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself',
'yourselves', 'the'])

class InvertedIndex:
    def __init__(self):
        pass

    def word_split(self,text):
        """
        Split a text in words. Returns a list of tuple that contains
        (word, location) location is the starting byte position of the word.
        """
        word_list = []
        wcurrent = []
        windex = None

        for i, c in enumerate(text):
            if c.isalnum():
                wcurrent.append(c)
                windex = i
            elif wcurrent:
                word = u''.join(wcurrent)
                word_list.append((windex - len(word) + 1, word))
                wcurrent = []

        if wcurrent:
            word = u''.join(wcurrent)
            word_list.append((windex - len(word) + 1, word))

        return word_list

    def words_cleanup(self,words):
        """
        Remove words with length less then a minimum and stopwords.
        """
        cleaned_words = []
        for index, word in words:
            if len(word) < _WORD_MIN_LENGTH or word in _STOP_WORDS:
                continue
            cleaned_words.append((index, word))
        return cleaned_words

    def words_normalize(self,words):
        """
        Do a normalization precess on words. In this case is just a tolower(),
        but you can add accents stripping, convert to singular and so on...
        """
        normalized_words = []
        for index, word in words:
            wnormalized = word.lower()
            normalized_words.append((index, wnormalized))
        return normalized_words

    def word_index(self,text):
        """
        Just a helper method to process a text.
        It calls word split, normalize and cleanup.
        """
        words = self.word_split(text)
        words = self.words_normalize(words)
        words = self.words_cleanup(words)
        return words

    def inverted_index(self,text):
        """
        Create an Inverted-Index of the specified text document.
            {word:[locations]}
        """
        inverted = {}

        for index, word in self.word_index(text):
            locations = inverted.setdefault(word, [])
            locations.append(index)

        return inverted

    def inverted_index_add(self,inverted, doc_id, doc_index):
        """
        Add Invertd-Index doc_index of the document doc_id to the
        Multi-Document Inverted-Index (inverted),
        using doc_id as document identifier.
            {word:{doc_id:[locations]}}
        """
        for word, locations in doc_index.items():
            indices = inverted.setdefault(word, {})
            indices[doc_id] = locations
        return inverted

    def search(self,index_file, query):
        """
        Returns a set of documents id that contains all the words in your query.
        """
        inverted=pickle.load(open(index_file,"rb"))
        words = [word for _, word in self.word_index(query) if word in inverted]
        results = [set(inverted[word].keys()) for word in words]
        return reduce(lambda x, y: x & y, results) if results else []

    def create(self,index_file,inverted_index_file):
        list_id_text = pickle.load(open(index_file, 'rb'))
        # Build Inverted-Index for documents
        inverted = {}
        documents = {}
        for item in list_id_text:
            documents[item[0]] = item[1]

        for doc_id, text in documents.items():
            doc_index = self.inverted_index(text)
            self.inverted_index_add(inverted, doc_id, doc_index)

        pickle.dump(inverted,open(inverted_index_file,"wb"))

    def create_with_data(self,index_data,inverted_index_file):
        # Build Inverted-Index for documents
        inverted = {}
        documents = {}
        for item in index_data:
            documents[item[0]] = item[1]

        for doc_id, text in documents.items():
            doc_index = self.inverted_index(text)
            self.inverted_index_add(inverted, doc_id, doc_index)

        pickle.dump(inverted,open(inverted_index_file,"wb"))

    def test(self,inverted_index_file):
        import time
        # Print Inverted-Index
        # for word, doc_locations in inverted.items():
        #    print(word, doc_locations)
        inverted=pickle.load(open(inverted_index_file,"rb"))

        # Search something and print results
        start_time = time.time()
        queries = ['gram-negative bacteria', 'Fungal infection', 'purulent exudate']
        for query in queries:
            result_docs = self.search(inverted, query)
            print("Search for '%s': %r" % (query, result_docs))
            '''
            for _, word in word_index(query):
                def extract_text(doc, index):
                    return documents[doc][index:index+40].replace('\n', ' ')

                for doc in result_docs:
                    for index in inverted[word][doc]:
                        print('   - '+doc+' %s...' % extract_text(doc, index))
            print()
            '''
            print(result_docs)
        end_time = time.time()
        time_passed = round((end_time - start_time) * 1000, 2)
        print("time:", time_passed, "ms")
        print()








