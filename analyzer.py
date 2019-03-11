from sys import argv, stderr, exit
import re
from camel_tools.calima_star.database import CalimaStarDB
from camel_tools.calima_star.analyzer import CalimaStarAnalyzer
from camel_tools.utils.dediac import dediac_ar
import numpy as np
import os


DESEG_DIR = os.path.dirname(os.path.realpath(argv[0]))

### Compile orthographic normalization regular expressions
NORM_ALIF_RE = re.compile(r'[آأإٱ]')
NORM_YAA_RE = re.compile(r'ى')
NORM_SPECIAL_RE = re.compile(r'[/+_#]')
DIALECT_RE = re.compile(r'\[...\]')


class Analyzer:

    """
    This class should describe an analyzer that can take an input word
        and run it through some de-lexical grammar
    The get_possible_segmentations function should return a set of tiples 
        Each triple will represent a possible segmentation
            The first item in the triple is a potentially empty list of proclitics,
            The second item is the base, represented as a string
            The third item is a potentially empty list of enclitics
        ( [[proclitics1], base1, [enclitics1]], [[proclitic2], base2, [enclitics2]], ... )
    """

    def __init__(self, grammar, separator, min_base_length, dialects):

        self.separator = separator
        self.grammar_file = grammar
        self.min_base_length = min_base_length
        ### Dialects are represented as subgrammars that are merged to form the full grammar
            ## Each analysis produced by a grammar should report the subgrammar generating it
            ## These can then be pruned if they do not appear in the list of desired dialects here
        self.dialects = dialects
        ### The free built-in grammar database doesn't distinguish bases
            ## If you use this grammar, here's a cheap hack to predict the base token
        if self.grammar_file == 'built-in' or 'built-in' in self.dialects:
            ### The free grammar only supports the MSA variety of Arabic
            self.dialects = ['built-in', 'MSA']
            self.grammar = CalimaStarDB(os.path.join(DESEG_DIR, 'grammar.db'), 'a')
            ### Order of tags used to predict which token belongs to base when multiple tags occur
                ## I did this in 5 minutes as proof of concept.. the order could be improved
                ## If you really want good results on MSA, consider buying the Sama database
            self.open_classes_hierarchy = [
                'NOUN', 'ADJ', 'VERB', 'IV', 'PV', 'CV', 'ADV', 'NOUN_PROP', 'IV_PASS', 'PV_PASS',
                'VERB_PART', 'FOREIGN', 'PSEUDO_VERB', 'FOCUS_PART', 'REL_ADV', 'ABBREV',  'PART',
                'INTERROG_PRON', 'REL_PRON', 'NOUN_QUANT', 'PRON_3MS', 'PRON_3MP', 'PRON_3D',
                'PRON_2D' 'PRON_2MS', 'PRON_2FS', 'PRON_1S', 'PRON_2MS', 'PRON_2MP', 'PRON_3FS',
                'PRON_3FP', 'PRON_2D', 'PRON_1P',  'DEM_PRON_FP', 'DEM_PRON_MP', 'DEM_PRON_MS',
                'DEM_PRON', 'DEM_PRON_F', 'DEM_PRON_FD', 'DEM_PRON_MD', 'DEM_PRON_FS', 'FUT_PART',
                'NEG_PART', 'VOC_PART', 'NOUN_NUM', 'PREP', 'SUB_CONJ', 'CONJ', 'INTERJ',
                'INTERROG_ADV', 'INTERROG_PART', 'EXCLAM_PRON', 'NUMERIC_COMMA', 'PUNC', 'DET']

        else:
            ### Try to load the specified grammar database in analyze mode
            try:
                self.grammar = CalimaStarDB(grammar, 'a')
            ### Resort to the free built-in grammar database if the specified database can't be found
            except FileNotFoundError:
                stderr.write('\nCould not locate grammar database "{}"\nLoading built-in database almor-msa\n'.format(grammar))
                self.grammar = CalimaStarDB(os.path.join(DESEG_DIR, 'grammar.db'), 'a')
                self.grammar_file = 'built-in'
                self.dialects = ['built-in', 'MSA']

        ### Run the analyzer in back-off mode, where input words can be any POS
        self.analyzer = CalimaStarAnalyzer(self.grammar, 'NOAN_ALL')


    def get_possible_segmentations(self, word):

        ### Assumes input word is already normalized if necessary
        possible_segmentations = []
        min_base_length = min(len(word), self.min_base_length)

        ### Run the analyzer
        try:
            analyses = self.analyzer.analyze(word)
            completed_analyses = {}

            ### Parse each analysis
            for analysis in analyses:

                ### Check the subgrammar that produced it
                dialect = self.get_analysis_dialect(analysis)
                if dialect in self.dialects:

                    possible_segmentation = [[], None, [], dialect]
                    ### Parse free built-in Almor grammar analysis
                    if 'built-in' in self.dialects:
                        analysis = dediacritize_normalize(self.accomodate_built_in_grammar(word, analysis))
                    ### Parse non-standard dialect subgrammar analysis
                    elif dialect != 'MSA':
                        analysis = dediacritize_normalize(self.accomodate_DA_grammar(word, analysis))
                    ### Parse Sama MSA grammar analysis
                    else:
                        analysis = dediacritize_normalize(analysis.get('d3seg', None))
                    
                    ### If no analysis, default to the entire word as the base
                    if analysis == None:
                        possible_segmentation[1] = word
                        possible_segmentations.append(possible_segmentation)
                        break

                    ### Make sure no segmentations leak into the segmentations
                        ## (our grammars are adapted from databases designed for segmentation)
                    if tuple([analysis, dialect]) not in completed_analyses:
                        completed_analyses[tuple([analysis, dialect])] = True
                        cat_tok = analysis.replace('+','').replace('_','')
                        if cat_tok == word:

                            ### Separate tokens
                            analysis = analysis.split('_')
                            ### Handle words entirely consisting of diacritics
                            if len(analysis) == 0:
                                possible_segmentation[1] = word
                            ### For non-empty words
                            else:
                                ### Only consider tokens consisting of more than just diacritics
                                all_tokens_empty = True
                                for token in analysis:
                                    if len(token.strip(self.separator)) != 0:
                                        all_tokens_empty = False

                                        ### handle proclitics
                                        if self.separator == token[-1]:
                                            possible_segmentation[0].append(token)
                                        ### handle enclitics
                                        elif self.separator == token[0]:
                                            possible_segmentation[2].append(token)
                                        ### handle base
                                        else:
                                            possible_segmentation[1] = token
                                ### Finish handling words entirely consisting of diacritics
                                if all_tokens_empty:
                                    possible_segmentation[1] = word

                            ### Prune ill-formed bases
                            base = possible_segmentation[1]
                            if base != None and len(base) >= min_base_length and possible_segmentation not in possible_segmentations: # and base in self.vocabulary
                                possible_segmentations.append(possible_segmentation)

        ### If inconsistency in the database, word will be the base with no clitics
        except KeyError:
            possible_segmentation = [[], word, [], 'MSA']
            possible_segmentations.append(possible_segmentation)
            stderr.write('\nGrammar database key error for {}\nUsing default segmentation analysis {}\n'.format(word, str(possible_segmentations)))

        ### And if no reasonable analyses are produced, default base is the word with no clitics
        if len(possible_segmentations) == 0:
            possible_segmentations = [[[], word, [], 'MSA']]

        return possible_segmentations


    def accomodate_DA_grammar(self, word, analysis):

        ### DA doesn't give D3tok so we need to parse diac
        analysis_seg = analysis['diac'].replace('_', '+').split('#')

        if len(analysis_seg) != 3:
            stderr.write('Bad Analysis!!!\n\t{}\n{}\n{}\n\n'.format(str(analysis_seg), str(word), str(analysis)))
            analysis_seg = ['', word, '']

        tokens = []

        proclitics = analysis_seg[0].split('+')
        for pro in proclitics:
            tokens.append('{}+_'.format(pro))

        tokens.append(analysis_seg[1])

        enclitics = analysis_seg[2].split('+')
        for en in enclitics:
            tokens.append('_+{}'.format(en))
                
        return ''.join(tokens)


    def accomodate_built_in_grammar(self, word, analysis):

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
            stderr.write('Could not find a base token!\nPlease add the problematic tag to the open_classes_hierarchy in the greedy_analyzer.py')
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


    def get_analysis_dialect(self, analysis_dict):

        if 'built-in' in self.dialects:
            return 'MSA'
        else:
            dialect = DIALECT_RE.findall(analysis_dict['gloss'])
            if len(dialect) == 0:
                return None
            else:
                return dialect[0][1:4]


##### ORTHOGRAPHIC NORMALIZATION REGULAR EXPRESSION FUNCTIONS #####
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

    grammar = argv[1]
    input_file = argv[2]
    grammar_accomodation = None 
    if len(argv) > 3:
        grammar_accomodation = argv[3:]

    analyzer = Analyzer(grammar, '+', 3, grammar_accomodation)
    if analyzer.grammar_file == 'built-in':
        grammar_accomodation = analyzer.grammar_file

    snum = 0
    words_to_possible_segmentations = {}
    for sent in open(input_file):
        snum += 1
        for word in sent.split():
            word = dediacritize_normalize(word)
            if word not in words_to_possible_segmentations:
                words_to_possible_segmentations[word] = analyzer.get_possible_segmentations(word)
        # print(snum)

    for word in words_to_possible_segmentations:
        print(word)
        for segmentation in words_to_possible_segmentations[word]:
            print('\tPRO: {}'.format(''.join(segmentation[0])))
            print('\tBASE: {}'.format(segmentation[1]))
            print('\tEN: {}'.format(''.join(segmentation[2])))
            print()









