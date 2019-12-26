import os
import copy
from parlai.core.teachers import FbDialogTeacher


def _path(opt, task_name):
    # Build the data if it doesn't exist.
    dt = opt['datatype'].split(':')[0]
    _file_path = os.path.join(opt['datapath'], 'wmt19_ende', task_name, dt + '.txt')
    assert os.path.exists(_file_path), 'Your dataset {} does dot exist!'.format(_file_path)
    return _file_path


class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, opt.get('subtask'))
        super().__init__(opt, shared)
