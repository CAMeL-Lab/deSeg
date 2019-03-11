from sys import argv, stderr, stdout, exit
import os
from analyzer import Analyzer, dediacritize_normalize, replace_special_characters
import pickle
import argparse
import multiprocessing as mp

DESEG_DIR = os.path.dirname(os.path.realpath(argv[0]))

class Disambiguator():

    def __init__(self, analyzer, separator, dialects):

        self.analyzer = analyzer
        self.separator = separator
        self.word_2_possible_segmentations = {}
        self.word_2_best_segmentation = {}
        self.dialects = dialects


    ##### TRAIN MODE FUNCTIONS #####
    def get_possible_segmentation_statistics(self, input_file, threads):
    
        ### Get vocabulary
        self.vocabulary = {}
        for sentence in open(input_file):
            for word in sentence.split():
                ## Alif-yaa normalize spelling and replace special characters with a dash
                normalized_word = replace_special_characters(dediacritize_normalize(word))
                if normalized_word not in self.vocabulary:
                    self.vocabulary[normalized_word] = 0
                self.vocabulary[normalized_word] += 1
        self.analyzer.vocabulary = self.vocabulary
        self.min_base_length = self.analyzer.min_base_length

        ### Check to see if the vocabulary has already been analyzed by the desired analyzer
        words_file = '{}.words.pkl'.format(os.path.basename(input_file))
        analyses_file = '{}.analyses_{}.pkl'.format(os.path.basename(input_file), os.path.basename(self.analyzer.grammar_file))
        try:
            normalized_analyses = pickleIn(analyses_file)
            stderr.write('\nLoading cached vocabulary and analyses\n\t{}\n'.format('\n\t'.join([words_file, analyses_file])))
            normalized_words = pickleIn(words_file)

        ### If it hasn't, prepare to analyze the vocabulary
        except FileNotFoundError:
            ## Unless the corpus specifies a dialect, consider all dialects in the initial analyses
                # Faster to do this and prune later than analyze from scratch for each dialect
            self.analyzer.dialects = ['BEI', 'CAI', 'ABD', 'RAB', 'TUN', 'MSA']
            if 'built-in' in self.dialects:
                self.analyzer.dialects.append('built-in')
            preprune = False
            for city in ['Beirut', 'Cairo', 'Doha', 'Rabat', 'Tunis', 'msa']:
                if city in input_file:
                    self.analyzer.dialects = self.dialects
                    preprune = True
                    break
            if preprune:
                stderr.write('\nPre-pruning analyses for dialects {}\n\n'.format(', '.join(self.dialects)))
            else:
                stderr.write('\nGettting all analyses for dialects {}\n\tWill prune after..\n\n'.format(', '.join(self.dialects)))
            ## Same with the minimum base length feature: leave it at most inclusive (smallest) value
                # Then prune the analyses based on the actual min base length later
            self.analyzer.min_base_length = 2

            ### Analyze the vocabulary
            normalized_words, normalized_analyses = self.analyze_vocabulary(threads)

            ### Cache out the analyzed vocabulary
            stderr.write('\nCaching out analyses: \n\t{}\n'.format('\n\t'.join([words_file, analyses_file])))
            pickleOut(normalized_words, words_file)
            pickleOut(normalized_analyses, analyses_file)

            ### Return min base length and specified dialects to their original values
                ## to prepare for pruning of analyses
            self.analyzer.min_base_length = self.min_base_length
            self.analyzer.dialects = self.dialects

        ### Prune out of dialect analyses and analyses with too small of bases
        for w in range(len(normalized_analyses)):
            ## Minimum base length can be violated only if the word itself is shorter than min base
            min_base_length = min(self.min_base_length, len(normalized_words[w]))
            valid_analyses = []
            for t in range(len(normalized_analyses[w])-1, -1, -1):
                dialect = normalized_analyses[w][t][-1]
                if dialect in self.dialects and len(normalized_analyses[w][t][1]) >= min_base_length:
                    valid_analysis = normalized_analyses[w][t][0:3]
                    # Only store unique valid analyses
                    if valid_analysis not in valid_analyses:
                        valid_analyses.append(valid_analysis)
            ## Default to the word itself as the affix-less base if no analyses are produced
            if len(valid_analyses) == 0:
                valid_analyses.append([[], normalized_words[w], []])
            normalized_analyses[w] = list(valid_analyses)

        ### Calculate the fertility of every base posited in any analysis across the vocabulary
        self.base_fertilities = {}
        stderr.write('Ranking bases by fertility\n')
        for i in range(len(normalized_analyses)):
            word = normalized_words[i]
            possible_segmentations = normalized_analyses[i]
            self.count_fertility(word, possible_segmentations)
        self.base_fertilities['OOV'] = {'OOV':True}


    def analyze_vocabulary(self, threads):

        ### Get full vocabulary to analyze
        normalized_words = list(self.vocabulary)
        len_norm_words = len(normalized_words)
        stderr.write('\nAnalyzing {} words..\n'.format(str(len_norm_words)))

        ### Chunk the vocabulary to be analyzed in parallel and track progress
        global progress
        progress = 0
        chunksize = int((len_norm_words/threads) + 1)
        stderr.write('Each of {} threads must complete {} words\n'.format(str(threads), str(chunksize)))
        with mp.Pool(threads) as p:
            ## analyze_parallel_pooling function analyzes 1000-word chunks from normalized_words list
            normalized_analyses = p.map(self.analyze_parallel_pooling, normalized_words, chunksize=1000)

        return normalized_words, normalized_analyses


    def analyze_parallel_pooling(self, word):

        ### Track the progress of each thread and write out after each 1000-word chunk
        global progress
        progress += 1
        if progress % 1000 == 0:
            stderr.write('\tThread completed {} words completed\n'.format(str(progress)))
            stderr.flush()
        ### Get the possible segmentations for each word in the chunk
        possible_segmentations = self.analyzer.get_possible_segmentations(word)

        return possible_segmentations


    def count_fertility(self, word, possible_segmentations):

        ### Infer bases and adjacent morphemes
            ## Full exponence is less relevant than what can immediately connect with base
            ## As then the fertility of exponent members cannot be separated from fertility of base
        self.word_2_possible_segmentations[word] = []       
        for possible_segmentation in possible_segmentations:
            ### Get base
            base = possible_segmentation[1]
            ### Get preceding morpheme
            if len(possible_segmentation[0]) > 0:
                pre = possible_segmentation[0][-1]
            else:
                pre = ''
            ### Get following morpheme
            if len(possible_segmentation[2]) > 0:
                post = possible_segmentation[2][0]
            else:
                post = ''
            ### Combine pre/postceding to get the adjacent exponence
            exponence = '{}|{}'.format(pre, post)

            ### Record all adjacent exponences for each base
            if base not in self.base_fertilities:
                self.base_fertilities[base] = {}
            self.base_fertilities[base][exponence] = True

            ### save base, adjacent exponence, and the possible_segmentation
            self.word_2_possible_segmentations[word].append([base, exponence, possible_segmentation])


    ##### APPLY MODE FUNCTIONS #####
    def apply_segmentation(self, input_file, output_file, priority=None, debug=False):

        ### Go through corpus and segment each sentence
        output = open(output_file, 'w')
        for sentence in open(input_file):
            output.write('{}\n'.format(' '.join(self.segment_sentence(sentence, priority=priority, debug=debug))))
        output.close()


    def segment_sentence(self, sentence, priority=None, debug=False):

        ### Go through each sentence and segment each word
        segmented_sentence = []
        for word in sentence.split():
            ## Normalize word and replace any special characters with a dash
            word = replace_special_characters(dediacritize_normalize(word))
            ## Segment every type that has yet to be segmented
            if word not in self.word_2_best_segmentation:
                self.get_best_segmentation_per_word(word, priority=priority, debug=debug)
            segmentation = self.word_2_best_segmentation[word][0][-1]
            # Record any proclitics
            for proclitic in segmentation[0]:
                segmented_sentence.append(proclitic)
            # Record the base
            segmented_sentence.append(segmentation[1])
            # Record any enclitics
            for enclitic in segmentation[2]:
                segmented_sentence.append(enclitic)

        return segmented_sentence


    def get_best_segmentation_per_word(self, word, priority='fbc', debug=False):

        ### Rank best-to-worst segmentations for the given word
        self.word_2_best_segmentation[word] = []

        ### Get analyses for any Out Of Vocabulary words
        if word not in self.word_2_possible_segmentations:
            self.count_OOV(word)

        ### Go through all possible segmentations and get the following:
        for possible_segmentation in self.word_2_possible_segmentations[word]:
            ## base
            [base, exponence, segmentation] = possible_segmentation
            ## base fertility
            fertility = len(self.base_fertilities[base])
            ## frequency with which proposed base appears as a stand-alone type in the corpus
            if base not in self.vocabulary:
                count_base_as_type = 0.5
            else:
                count_base_as_type = self.vocabulary[base]
            ## base length
            real_base = segmentation[1]
            base_length = len(real_base)

            ### Rank fertility, base length, and frequency based on pre-specified priorities
                ## In general, we want analyses with
                    # high base fertilities
                    # small stems (because we're maximum matching on known possible exponents)
                    # high base-as-type frequencies
            if priority == 'fbc':
                self.word_2_best_segmentation[word].append([fertility, -base_length, count_base_as_type, segmentation])
            elif priority == 'fcb':
                self.word_2_best_segmentation[word].append([fertility, count_base_as_type, -base_length, segmentation])
            elif priority == 'bfc':
                self.word_2_best_segmentation[word].append([-base_length, fertility, count_base_as_type, segmentation])
            elif priority == 'bcf':
                self.word_2_best_segmentation[word].append([-base_length, count_base_as_type, fertility, segmentation])
            elif priority == 'cfb':
                self.word_2_best_segmentation[word].append([count_base_as_type, fertility, -base_length, segmentation])
            elif priority == 'cbf':
                self.word_2_best_segmentation[word].append([count_base_as_type, -base_length, fertility, segmentation])
            else:
                stderr.write('\nUNSUPPORTED RANKING PRIORITY {}\n\n'.format(priority))
                exit()

        ### Rank segmentations given priorities over fertility maximization, base minimization, and frequency maximization
        self.word_2_best_segmentation[word].sort(reverse=True)

        ### There should always be at least one proposed segmentation
        if len(self.word_2_best_segmentation[word]) == 0:
            stderr.write('NO SEGMENTATIONS FOUND!!!{}\n'.format('\n\t'.join(str(x) for x in [word, word_2_possible_segmentations[word]])))
            exit()

        ### Print ranking and proposed base fertilities for debug/development purposes
        if debug:
            stderr.write('\n{}\n'.format(word))
            for segmentation in self.word_2_best_segmentation[word]:
                stderr.write('\t{}\n'.format(segmentation[-1]))
                stderr.write('\t\t{}\n'.format(', '.join(str(x) for x in segmentation[0:3])))
                base = segmentation[-1][1]
                if base not in self.base_fertilities:
                    base = 'OOV'
                stderr.write('\t\tBase_fertilities: {}\n'.format(' -- '.join(list(str(x) for x in (list(self.base_fertilities[base].keys()))))))


    def count_OOV(self, word):

        ### Run analyzer to get possible segmentations
        self.word_2_possible_segmentations[word] = []  
        possible_segmentations = self.analyzer.get_possible_segmentations(word)

        ### We ran the analyzer in pre-prune mode, so we know the dialect will be valid
        for possible_segmentation in possible_segmentations:
            possible_segmentation = possible_segmentation[:-1]
            ## Get the base of each analysis
            base = possible_segmentation[1]
            ## Get any proclitics
            if len(possible_segmentation[0]) > 0:
                pre = possible_segmentation[0][-1]
            else:
                pre = ''
            ## Get any enclitics
            if len(possible_segmentation[2]) > 0:
                post = possible_segmentation[2][0]
            else:
                post = ''
            ## Get the adjacent exponenct
            exponence = tuple('{}|{}'.format(pre, post))

            ## Determine proposed base fertility
            if base not in self.base_fertilities:
                base = 'OOV'

            ## Record segmentation analysis
            to_add = [base, exponence, possible_segmentation]
            if to_add not in self.word_2_possible_segmentations[word]:
                self.word_2_possible_segmentations[word].append(to_add)


    ## PRINT FUNCTIONS ##
    def print_most_frequent_tokens(self):

        token_classes = ['pre', 'base', 'post']
        class_2_token_frequency = {x:{} for x in token_classes}
        for word in self.word_2_best_segmentation:
            for i in range(len(token_classes)):
                token_class = token_classes[i]
                if token_class == 'base':
                    base = self.word_2_best_segmentation[word][0][-1][i]
                    if base not in class_2_token_frequency[token_class]:
                        class_2_token_frequency[token_class][base] = 0
                    class_2_token_frequency[token_class][base] += 1
                else:
                    if len(self.word_2_best_segmentation[word][0][-1][i]) == 0:
                        token = ''
                        if token not in class_2_token_frequency[token_class]:
                            class_2_token_frequency[token_class][token] = 0
                        class_2_token_frequency[token_class][token] += 1
                    for token in self.word_2_best_segmentation[word][0][-1][i]:
                        if token not in class_2_token_frequency[token_class]:
                            class_2_token_frequency[token_class][token] = 0
                        class_2_token_frequency[token_class][token] += 1

        for token_class in token_classes:
            ranked_list = [[class_2_token_frequency[token_class][x], x] for x in class_2_token_frequency[token_class]]
            ranked_list.sort(reverse=True)
            stdout.write('MOST FREQUENT {} TOKENS\n'.format(token_class))
            for l in ranked_list:
                stdout.write('\t{} -- {}\n'.format(l[1], str(l[0])))


    def print_ranked_segmentations_by_word(self):

        stdout.write('RANKED SEGMENTATIONS BY WORD\n')
        for word in self.word_2_best_segmentation:
            stdout.write('{}\n'.format(word))
            for segmentation in self.word_2_best_segmentation[word]:
                ranked_factors = segmentation[0:3]
                segmentation = segmentation[-1]
                seg = []
                for pro in segmentation[0]:
                    seg.append(pro)
                seg.append(segmentation[1])
                for en in segmentation[2]:
                    seg.append(en)
                stdout.write('\t{}\n'.format(' '.join(seg)))
                stdout.write('\t\t{}\n'.format(' -- '.join(str(x) for x in ranked_factors)))


    ##### INTERACTIVE MODE FUNCTIONS #####
    def interact(self, debug=False, priority=None):

        ### Run interactive mode until the user quits
        ## Basically, this just runs apply mode on raw input
        os.system('clear')
        stdout.write('Welcome to interactive mode!\nYou can enter q at any time to quit.\n\nSo like what do you wanna segment or whatever?\n\n: ')
        while True:
            sentence = input()
            if sentence in ['q', 'Q']:
                exit()
            stdout.write('{}\n\n\n\n: '.format(' '.join(self.segment_sentence(sentence, debug=debug, priority=priority))))


##### GENERAL FUNCTIONS #####
def str2bool(v):

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_disambiguator_filename(args):

    ### Get the name of the file to cache the relevant disambiguator model in
    disambiguator_file = args.cached_disambiguator
    if disambiguator_file == None:
        disambiguator_file = 'disambiguator.{}.{}.minBase{}.pkl'.format(os.path.basename(args.train), os.path.basename(args.grammar), str(args.min_base_length))
        disambiguator_file = os.path.join(DESEG_DIR, disambiguator_file)

    return disambiguator_file


def pickleIn(file):

    pklFile = open(file, 'rb')
    ofTheKing =pickle.load(pklFile)
    pklFile.close()

    return ofTheKing


def pickleOut(thingy, file):

    pklFile = open(file, 'wb')
    pickle.dump(thingy, pklFile)
    pklFile.close()


#########################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'apply', 'interactive'], help='You can run the deSeg segmenter in either "train", "interactive", or "apply" mode', required=True)
    parser.add_argument('-t', '--train', type=str, help='Location of the corpus from which we will learn the statistics necessary to apply the deSeg segmenter', required=False, default=None)
    parser.add_argument('-T', '--test', type=str, help='Location of the corpus we will apply the learned segmenter to', required=False, default=None)
    parser.add_argument('-o', '--output', type=str, help='Location where the segmented corpus will be written out to', required=False, default='output.tok')
    parser.add_argument('-g', '--grammar', type=str, help='Grammar to be used by the analyzer', required=False, default='built-in')
    parser.add_argument('-a', '--accomodated_grammars', nargs='+', choices=['built-in', 'MSA', 'RAB', 'TUN', 'BEI', 'CAI', 'ABD'], help="Filters analyses by those produced by the relevant dialects' grammars", required=False, default=['MSA'])
    parser.add_argument('-c', '--cached_disambiguator', type=str, help='Where to load or store the trained disambiguator', required=False, default=None)
    parser.add_argument('-s', '--separator', type=str, help='Separator used to mark token boundaries', required=False, default='+')
    parser.add_argument('-l', '--min_base_length', type=int, help='Minimum length of the base (remainder after separating exponent tokens) for the analysis to be considered feasible.', required=False, default=1)
    parser.add_argument('-M', '--multi_thread', type=int, help='Number of threads for parallel computing.', required=False, default=12)
    parser.add_argument('-p', '--priority', type=str, choices=['fbc', 'fcb', 'bfc', 'bcf', 'cfb', 'cbf'], help="Priority over fertility of base (f), base length (b), and base token count (c) when choosing a candidate segmentation. Order in which the factors are listed represents their priority during evaluation.", required=False, default='fbc')
    parser.add_argument('-P', '--print_options', nargs='+', help="Optional print statements that can be executed for debugging/development purposes. They report the most (token) frequent proclictics, enclitics, and bases chosen by the disambiguator and/or a ranking for each word of the disambiguator's preferences over the analyzer's proposed segmentations.", required=False, choices=['most_frequent_tokens', 'ranked_segmentations_by_word'], default=[])
    parser.add_argument('-d', '--debug', type=str2bool, help="Compute segmentations in debug mode.", required=False, default=False)

    args = parser.parse_args()

    ### Since MSA is the high register of all dialects, make sure we consider these analyses
    if 'MSA' not in args.accomodated_grammars:
        args.accomodated_grammars.append('MSA')

    ### TRAINING MODE
    if args.mode == 'train':

        # Initialize the analyzer
        stderr.write('\nInitializing analyzer with grammar "{}"..\n'.format(args.grammar))
        analyzer = Analyzer(args.grammar, args.separator, args.min_base_length, args.accomodated_grammars)

        # Built-in grammar only accomodates MSA analyses
        if analyzer.grammar_file == 'built-in':
            args.accomodated_grammars = ['built-in', 'MSA']

        # Train the disambiguator
        disambiguator_file = get_disambiguator_filename(args)
        try:
            disambiguator = pickleIn(disambiguator_file)
            command = 'python greedy_disambiguator.py -m apply -c {} -T [data_to_segment] -o [desired_output_file]'.format(disambiguator_file)
            stderr.write('\nDisambiguator "{}" has already been trained!\nTo apply the disambiguator, run the following command:\n{}\n'.format(disambiguator_file, command))
            exit()
        except FileNotFoundError:
            stderr.write('\nTraining disambiguator on "{}"..\n'.format(args.train))
            disambiguator = Disambiguator(analyzer, args.separator, args.accomodated_grammars)
            disambiguator.get_possible_segmentation_statistics(args.train, args.multi_thread)

        # Save the trained disambiguator    
        stderr.write('\nCaching trained disambiguator..\n')
        pickleOut(disambiguator, disambiguator_file)

        # Move on to bigger and better things
        if len(args.print_options) > 0:
            stderr.write('\nThe print_options can only be used in Apply mode. You can apply the trained disambiguator to "{}" though to print any relevant statistics from the training set.'.format(args.train))
        stderr.write('\nDone! _\n     / \\ \n    /   \\    \n __/_____\\__\n   /     \\\n  | \\   / |\n| |   v   | |\n| |  ___  | |\n|  \\_____/  |\n \\____|____/\n      |\n      |\n      |\n      |\n      |\n     / \\\n    /   \\\
        \n   /     \\\n  |       |\n  |       |\n  |       |\nSo yeah, your trained segmenter is here: "{}"\n\n'.format(disambiguator_file))


    # Read in the trained disambiguator
    else:
        try:
            stderr.write('\nReading in the pre-trained disambiguator "{}"\n'.format(args.cached_disambiguator))
            disambiguator = pickleIn(args.cached_disambiguator)
        except FileNotFoundError:
            stderr.write('\tDisambiguator not found!!!\n'.format(args.cached_disambiguator))
            exit()


    ### INTERACTIVE MODE
    if args.mode == 'interactive':

        disambiguator.interact(debug=args.debug, priority=args.priority)


    ### APPLY SEGMENTATION MODE
    elif args.mode == 'apply':

        # Apply the trained disambiguator and write out segmentation
        stderr.write('\nApplying segmentation.. reading input from "{}" and writing output to: "{}"..\n'.format(args.test, args.output))
        disambiguator.apply_segmentation(args.test, args.output, priority=args.priority, debug=args.debug)

        # Print out any requested information
        if 'most_frequent_tokens' in args.print_options:
            disambiguator.print_most_frequent_tokens()
        if 'ranked_segmentations_by_word' in args.print_options:
            disambiguator.print_ranked_segmentations_by_word()
        if len(args.print_options) == 0:
            stderr.write('\n')
            os.system('clear')

        # Shake your tailfeather
        stderr.write('Done! _\n     / \\ \n    /   \\    \n __/_____\\__\n   /     \\\n  | o   o |\n| |   v   | |\n| | \\___/ | |\n|  \\_____/  |\n \\____|____/\n      |\n      |\n      |\n      |\n      |\n     / \\\n    /   \\\
        \n   /     \\\n  |       |\n  |       |\n  |       |\nSo yeah, your segmented output is here: "{}"\n\n'.format(args.output))
