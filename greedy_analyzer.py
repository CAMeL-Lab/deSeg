from sys import argv, stderr, exit
import re
from camel_tools.calima_star.database import CalimaStarDB
from camel_tools.calima_star.analyzer import CalimaStarAnalyzer
from camel_tools.utils.dediac import dediac_ar


NORM_ALIF_RE = re.compile(r'[آأإٱ]')
NORM_YAA_RE = re.compile(r'ى')
NORM_SPECIAL_RE = re.compile(r'[/+_]')


class Analyzer:

    """
    This class should describe an analyzer that can take an input word
        and run it through Mai's greedy tokenizer
    The get_possible_tokenizations function should return a set of tiples 
    Each triple will represent a possible tokenization
    The first item in the triple is a potentially empty list of proclitics,
    The second item is the base, represented as a string
    The third item is a potentially empty list of enclitics
    ( [[proclitics1], base1, [enclitics1]], [[proclitic2], base2, [enclitics2]], ... )

    Mai's version of the get_possible_tokenizations function may also need arguments like
        proclitics_list, enclitics_list, forbidden_clitic_combos_list
    Just make sure wherever this function is called in greedy_disambiguator.py,
        that you add the additional arguments

    The code below is just a placeholder levaraging the SAMA MSA analyzer
        It's simply for debugging/benchmarking on MSA where we have lots of gold data
    """

    def __init__(self, database, separator):

        self.separator = separator
        self.database_file = database
        ### The free database does not distinguish base from affixal tokens in D3tok
            ## If you use the free database, here's a cheap hack to predict the base token
        if self.database_file == 'built-in':
            self.database = CalimaStarDB.builtin_db('almor-msa', 'a')
            ### Order of tags used to predict which belongs to base when multiple tags occur
                ## I did this in 5 minutes as proof of concept.. the order could be improved
                ## If you really want good results on MSA, you should just buy the Sama database
            self.open_classes_hierarchy = [
                'NOUN', 'ADJ', 'VERB', 'IV', 'PV', 'CV', 'ADV', 'NOUN_PROP', 'IV_PASS', 'PV_PASS',
                'VERB_PART', 'FOREIGN', 'PSEUDO_VERB', 'FOCUS_PART', 'REL_ADV', 'ABBREV',  'PART',
                'INTERROG_PRON', 'REL_PRON', 'NOUN_QUANT', 'PRON_3MS', 'PRON_3MP', 'PRON_3D',
                'PRON_2D' 'PRON_2MS', 'PRON_2FS', 'PRON_1S', 'PRON_2MS', 'PRON_2MP', 'PRON_3FS',
                'PRON_3FP', 'PRON_2D', 'PRON_1P',  'DEM_PRON_MP', 'DEM_PRON_MS', 'DEM_PRON',
                'DEM_PRON_F', 'DEM_PRON_FD', 'DEM_PRON_MD', 'DEM_PRON_FS', 'FUT_PART', 'NEG_PART',
                'VOC_PART', 'NOUN_NUM', 'PREP', 'SUB_CONJ', 'CONJ', 'INTERJ', 'INTERROG_ADV',
                'INTERROG_PART', 'EXCLAM_PRON', 'NUMERIC_COMMA', 'PUNC', 'DET']
        else:
            try:
                self.database = CalimaStarDB(database, 'a')
            except FileNotFoundError:
                stderr.write('\nCould not locate database {}\nLoading built in database almor-msa\n'.format(database))
                self.database = CalimaStarDB.builtin_db('almor-msa', 'a')
        self.analyzer = CalimaStarAnalyzer(self.database, 'NOAN_PROP')


    def accomodate_built_in_database(self, word, analysis):

        ### Almor doesn't give D3tok so we need to parse BW
        analysis = analysis['bw'].replace('+','/').strip('/').split('/')

        open_class_tag = None
        for open_class in self.open_classes_hierarchy:
            if open_class in analysis:
                open_class_tag = open_class
                break

        try:
            assert open_class_tag != None
        except:
            stderr.write('Could not find a base token!\n')
            stderr.write('{}\n'.format(word))
            stderr.write('{}\n'.format(str(analysis)))
            stderr.write('{}\n'.format(str(self.open_classes_hierarchy)))
            exit()

        try:
            assert len(analysis) % 2 == 0
        except:
            stderr.write('Malformed analysis!\n')
            stderr.write('{}\n'.format(word))
            stderr.write('{}\n'.format(str(analysis)))
            exit()

        tokens = []
        pro = True
        for m in range(0, len(analysis), 2):
            token = dediacritize_normalize(analysis[m])
            if len(token) > 0:
                if pro and analysis[m+1] == open_class_tag:
                    pro = False
                    tokens.append('{}'.format(token))
                else:
                    if pro:
                        tokens.append('{}+_'.format(token))
                    else:
                        tokens.append('_+{}'.format(token))
                
        return ''.join(tokens)


    def get_possible_tokenizations(self, word):

        ### assumes word is already dediacritized alif-yaa normalized
        possible_tokenizations = []

        ### Run the analyzer
        try:
            analyses = self.analyzer.analyze(word)
            completed_analyses = {}

            ### For each analysis
            for analysis in analyses:

                possible_tokenization = [[], None, []]

                ### Parse Almor database analysis
                if self.database_file == 'built-in':
                    analysis = self.accomodate_built_in_database(word, analysis)

                ### Parse Sama database analysis
                else:
                    analysis = analysis['d3tok']
                
                ### If no analysis, put the entire word as the base
                if analysis == None:
                    possible_tokenization[1] = word
                    possible_tokenizations.append(possible_tokenization)
                    break

                ### Dediacritize and Alif-Yaa normalize the analysis
                analysis = dediacritize_normalize(analysis)

                ### Prevent from doing the same tokenization multiple times
                if analysis not in completed_analyses:
                    completed_analyses[analysis] = True
                    
                    ### Separate tokens
                    analysis = analysis.split('_')
                    ### Handle words entirely consisting of diacritics
                    if len(analysis) == 0:
                        possible_tokenization[1] = word
                    ### For non-empty words
                    else:
                        ### Only consider tokens consisting of more than just diacritics
                        all_tokens_empty = True
                        for token in analysis:
                            if len(token.strip(self.separator)) != 0:
                                all_tokens_empty = False

                                ### handle proclitics
                                if self.separator == token[-1]:
                                    possible_tokenization[0].append('{}'.format(token))
                                ### handle enclitics
                                elif self.separator == token[0]:
                                    possible_tokenization[2].append('{}'.format(token))
                                ### handle base
                                else:
                                    possible_tokenization[1] = token
                        ### Finish andling words entirely consisting of diacritics
                        if all_tokens_empty:
                            possible_tokenization[1] = word

                    ### Exception handling for database errors
                        # 3lY as a proclitic
                        # it shouldn't be possible to have an empty base
                    good_analysis = True
                    if possible_tokenization[1] == None or len(possible_tokenization[1]) == 0:
                        ## 3lY should not be a proclitic.. database bug
                        if ['+علي'] == possible_tokenization[0][0]:
                            good_analysis = False
                        ## empty base
                        else:
                            possible_tokenization[1] = word

                    ### Add good tokenization analyses to the set of possible tokenizations
                    if good_analysis:
                        if possible_tokenization not in possible_tokenizations:
                            possible_tokenizations.append(possible_tokenization)


        ### If inconsistency in the database, word will be the base with no clitics
        except KeyError:
            possible_tokenization = [[], word, []]
            possible_tokenizations.append(possible_tokenization)
            stderr.write('\nDatabase key error for {}\nUsing default tokenization analysis {}\n'.format(word, str(possible_tokenizations)))

        ### And if no reasonable analyses are produced, default base is the word with no clitics
        if len(possible_tokenizations) == 0:
            possible_tokenizations = [[[], word, []]]

        return possible_tokenizations


def dediacritize_normalize(word):

    ### Dediacritize
    word = dediac_ar(word)
    ### Alif normalize
    word = NORM_ALIF_RE.sub('ا', word)
    ### Yaa normalize
    word = NORM_YAA_RE.sub('ي', word)

    return word

def replace_special_characters(word):

    ### Normalize special characters
    word = NORM_SPECIAL_RE.sub('-', word)

    return word


#########################################################################

if __name__ == '__main__':

    analyzer = Analyzer(argv[1], '+')

    words_to_possible_tokenizations = {}
    for sent in open(argv[2]):
        for word in sent.split():
            word = dediacritize_normalize(word)
            if word not in words_to_possible_tokenizations:
                words_to_possible_tokenizations[word] = analyzer.get_possible_tokenizations(word)
