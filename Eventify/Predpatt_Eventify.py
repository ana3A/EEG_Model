import spacy
from predpatt import PredPattOpts

from predpatt import PredPatt
from predpatt import load_conllu

import stanza
from predpatt.util.ud import dep_v2
from stanza.utils.conll import CoNLL

# Aux function for sorting
from Dataset.Utils.DataCleaner import DataCleaner


def comp(token):
    return token.position


class Eventify:
    subject_object = 'sv'
    subject_deps = ['nsubj', 'nsubj:pass', 'csubj', 'csubj:pass', 'conj']
    object_deps = ['obj', 'iobj', 'conj', 'ccomp']

    def __init__(self, resolve_relcl=False, resolve_appos=False, resolve_amod=False, resolve_conj=True,
                 resolve_poss=False, simple=True, cut=True, big_args=False, strip=True, ud=dep_v2.VERSION, mode="normal"):

        self.empty_token = '<|EMPTY|>'
        self.non_wanted_deps = ['aux', 'det', 'discourse', 'reparandum', 'mark', "aux:pass"]

        stanza.download('en')

        self.nlp = stanza.Pipeline('en',
                                    processors='tokenize,mwt,pos,lemma,depparse,ner', verbose=False,
                                    use_gpu=True)  # https://github.com/stanfordnlp/stanza

        self.opts = PredPattOpts(
            resolve_relcl=resolve_relcl,
            resolve_appos=resolve_appos,
            resolve_amod=resolve_amod,
            resolve_conj=resolve_conj,
            resolve_poss=resolve_poss,
            big_args=big_args,
            strip=strip,
            ud=ud,
            simple=simple,
            cut=cut
        )

        self.mode = mode
        self.stack = []

    def eventify(self, text):
        conll_sent, doc = self.generateConll(text)
        doc_events = []
        for sent_idx, conll in enumerate(conll_sent):
            conll_example = [ud_parse for sent_id, ud_parse in load_conllu(conll)][0]
            ppatt = PredPatt(conll_example, opts=self.opts)

            # Return events of the form (subject, neg + verb, object)
            id_to_pred = self.svoEvents(ppatt)

            for key in id_to_pred:
                pred_events = self.mountEvent(id_to_pred[key], doc, sent_idx)
                doc_events += pred_events

        return doc_events

    def mountEvent(self, event, doc, sent_idx):
        if event[0]:
            if self.mode != 'string':
                subjects = self.getSubjectString(event[0])
            else:
                subjects = self.getSubjectStanzaTokens(event[0], doc, sent_idx)
        else:
            subjects = [self.empty_token]

        if self.mode != 'string':
            predicate, prepositions = self.getActionString(event[1], doc, sent_idx)
        else:
            predicate, prepositions = self.getActionStanzaTokens(event[1], doc, sent_idx)

        if event[2]:
            if self.mode != 'string':
                objects = self.getObjectString(event[2], prepositions, doc, sent_idx)
            else:
                objects = self.getObjectStanzaTokens(event[2], prepositions, doc, sent_idx)
        else:
            objects = [self.empty_token]

        events = []
        for s in subjects:
            for o in objects:
                mounted_event = (s, predicate, o)
                events.append(mounted_event)

        return events

    def generateConll(self, text):
        doc = self.nlp(text)  # doc is class Document

        dicts = doc.to_dict()  # dicts is List[List[Dict]], representing each token / word in each sentence in the document
        conll_list = CoNLL.convert_dict(
            dicts)  # conll is List[List[List]], representing each token / word in each sentence in the document

        conll_sent = []
        for sentence in conll_list:
            sentence_conll = ''
            for word in sentence:
                sentence_conll += '\t'.join(conll_elem for conll_elem in word) + '\n'
            conll_sent.append(sentence_conll)
        return conll_sent, doc

    def svoEvents(self, ppatt, already_processed=None, begin=None, end=None):
        # ID to keep track of predicates. An ID is used to avoid having problems with predicates with the same text.
        id_to_pred = dict()

        if begin is None and end is None:
            begin = 0
            end = len(ppatt.instances)

        if already_processed is None:
            already_processed = []

        # For each predicate
        for i in range(begin, end):
            if i in already_processed:
                continue

            predicate = ppatt.instances[i]

            if self.isCop(predicate):
                continue

            action = predicate.tokens  # self.getAction(predicate, doc, sent_idx)

            pred_id = i
            id_to_pred[pred_id] = [None, action, None, None]

            passive = self.isPassiveVoice(predicate.arguments)

            # For each argument of the predicate
            for argument in predicate.arguments:
                arg = argument.tokens
                arg_name = argument.root.gov_rel
                arg_idx = None

                pred_number = self.isPredicate(argument, ppatt)
                if pred_number:
                    end = begin + 1
                    stack_index = 0
                    # Sometimes predicates refer to eachother in a loop in predpatt (rare)
                    if arg not in self.stack:
                        self.stack.append(arg)
                        stack_index = len(self.stack) -1
                    else:
                        continue

                    arg = [self.svoEvents(ppatt, begin=pred_number, end=pred_number + 1)]
                    self.stack.pop(stack_index)
                    already_processed.append(pred_number)
                    arg_idx = 2

                elif arg_name in Eventify.subject_deps:
                    arg_idx = 0 if not passive else 2

                elif arg_name in Eventify.object_deps:
                    arg_idx = 2 if not passive else 0

                elif 'conj' in arg_name:
                    for token in argument.tokens:
                        if token.gov_rel in Eventify.subject_deps:
                            arg_idx = 0 if not passive else 2
                        elif token.gov_rel in Eventify.object_deps:
                            arg_idx = 2 if not passive else 0

                if arg_idx is None:
                    continue

                if id_to_pred[pred_id][arg_idx] is None:
                    id_to_pred[pred_id][arg_idx] = [arg]

                else:
                    id_to_pred[pred_id][arg_idx].append(arg)


        return id_to_pred

    """
    Removes words that we want to remove from the predicate.
    """

    def wantTokenInPredicate(self, token):
        # Remove verb auxiliars

        if token.gov_rel in self.non_wanted_deps:
            return False

        # Remove adverbs that are not PART.
        # Why? Because the "not" has dep = advmod and upos = PART.
        if token.gov_rel == 'advmod' and not token.tag == 'PART':
            return False

        return token.isword

    def isPassiveVoice(self, arguments):
        for argument in arguments:
            arg_name = argument.root.gov_rel
            if 'pass' in arg_name:
                return True
        return False

    def isCop(self, p):
        for dep in p.root.dependents:
            if dep.rel == "cop":
                return True
        return False

    def isPredicate(self, argument, ppatt):
        index = 0
        for i, p in enumerate(ppatt.instances):
            if self.isCop(p):
                index -= 1
                continue
            if p.root == argument.root:
                index += i
                return index
        return None

    def getActionString(self, tokens, doc, sent_idx):
        prepositions = []
        action_tokens = []
        for token in tokens:
            # Prepositions and postpositions, together called adpositions (or broadly, in English, simply prepositions)
            # Universal POS tag = ADP
            if token.tag.lower() == 'adp' and token.gov_rel != "compound:prt":
                prepositions.append(token)
                continue

            if self.wantTokenInPredicate(token):
                action_tokens.append(token)

        action = " ".join(doc.sentences[sent_idx].words[token.position].lemma for token in action_tokens)

        return action, prepositions

    def getActionStanzaTokens(self, tokens, doc, sent_idx):
        prepositions = []
        for token in tokens:
            # Prepositions and postpositions, together called adpositions (or broadly, in English, simply prepositions)
            # Universal POS tag = ADP
            if token.tag.lower() == 'adp':
                prepositions.append(token)

        return [doc.sentences[sent_idx].words[token.position] for token in tokens], prepositions

    def getObjectString(self, objects_tokens, prepositions, doc, sent_idx):
        objs = []
        for o_t in objects_tokens:
            if isinstance(o_t[0], dict):
                for key in o_t[0]:
                    objs += self.mountEvent(o_t[0][key], doc, sent_idx)
            else:
                for prep in prepositions:
                    if prep.gov in o_t:
                        o_t.append(prep)
                o_t.sort(key=comp)
                objs.append(
                    " ".join(token.text for token in o_t if token.gov_rel not in self.non_wanted_deps and token.isword))

        return objs

    def getObjectStanzaTokens(self, objects_tokens, prepositions, doc, sent_idx):
        objs = []
        for o_t in objects_tokens:
            if isinstance(o_t[0], dict):
                for key in o_t[0]:
                    objs += self.mountEvent(o_t[0][key], doc, sent_idx)
            else:
                for prep in prepositions:
                    if prep.gov in o_t:
                        o_t.append(prep)
                o_t.sort(key=comp)
                objs.append([doc.sentences[sent_idx].words[token.position] for token in o_t])

        return objs

    def getSubjectString(self, subj_list):
        subjs = []
        for s in subj_list:
            subjs.append(
                " ".join(token.text for token in s if token.gov_rel not in self.non_wanted_deps and token.isword))
        return subjs

    def getSubjectStanzaTokens(self, subj_list, doc, sent_idx):
        subjs = []
        for s in subj_list:
            subjs.append(
                [doc.sentences[sent_idx].words[token.position] for token in s])
        return subjs




