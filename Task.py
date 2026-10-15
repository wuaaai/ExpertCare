from utils import get_init_tokenizers
from pyhealth.tokenizer import Tokenizer

def initialize_task(task_dataset, args):
    if args.task in ('drug_rec', 'drug_rec_ts'):
        Tokenizers_visit_event = get_init_tokenizers(task_dataset, keys=['conditions', 'procedures', 'drugs_hist'])
        Tokenizers_monitor_event = get_init_tokenizers(task_dataset,
                                                       keys=['lab_item', 'lab_flag', 'inj_item', 'inj_amt',
                                                             'lab_inj_groups', 'lab_inj_merged_list'])
        label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('drugs'))
        label_size = len(task_dataset.get_all_tokens('drugs'))
    elif args.task in ('diag_pred', 'diag_pred_ts'):
        Tokenizers_visit_event = get_init_tokenizers(task_dataset, keys=['cond_hist', 'procedures', 'drugs'])
        Tokenizers_monitor_event = get_init_tokenizers(task_dataset,
                                                       keys=['lab_item', 'lab_flag', 'inj_item', 'inj_amt'])
        label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens('conditions'))
        label_size = len(task_dataset.get_all_tokens('conditions'))
    else:
        raise ValueError('没有这个任务')

    return Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size
