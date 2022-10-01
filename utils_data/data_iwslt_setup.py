import gzip
import os
import shutil
import tarfile
import zipfile
from typing import Union

import torchtext


def process_iwslt(dataset_save_to, dataset_downloaded_in, languages: tuple = ("de", "en")):
    # TODO: check if files and folders exist
    files_to_move = []
    base_str_iwslt = dataset_downloaded_in + "/texts/"
    ll = [languages[0] + "-" + languages[1], languages[1] + "-" + languages[0]]

    l = [{"lang": languages[0], "other": languages[1]}, {"lang": languages[1], "other": languages[0]}]

    for language in l:
        file_name = language["lang"] + "-" + language["other"] + ".tgz"
        src_move_file = os.path.join(base_str_iwslt, language["lang"], language["other"], file_name)
        trg_move_file = os.path.join(dataset_save_to, file_name)
        print(f"Moving file: src_move_file {src_move_file} trg_move_file {trg_move_file}")
        shutil.copy(src_move_file, trg_move_file)


def extract(path, filename):
    zpath = os.path.join(path, filename)
    zroot, ext = os.path.splitext(zpath)
    _, ext_inner = os.path.splitext(zroot)
    if ext == '.zip':
        with zipfile.ZipFile(zpath, 'r') as zfile:
            print('extracting')
            zfile.extractall(path)
    # tarfile cannot handle bare .gz files
    elif ext == '.tgz' or ext == '.gz' and ext_inner == '.tar':
        with tarfile.open(zpath, 'r:gz') as tar:
            dirs = [member for member in tar.getmembers()]
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner) 
                
            
            safe_extract(tar, path=path, members=dirs)
    elif ext == '.gz':
        with gzip.open(zpath, 'rb') as gz:
            with open(zroot, 'wb') as uncompressed:
                shutil.copyfileobj(gz, uncompressed)
    return zpath, zroot


def exist_make(path: str):
    if os.path.exists(path):
        return
    else:
        os.makedirs(name=path)


def dataset_download_extract(dataset_name: str, dataset_path: Union[str, None] = None, dataset_root: str = ".data", project_root="../", dataset_url: str = "", overwrite: bool = False):
    save_path = os.path.join(project_root, dataset_root, dataset_name)
    print(os.path.abspath(save_path))
    exist_make(path=save_path)
    torchtext.utils.download_from_url(url=dataset_url, path=save_path, root=dataset_root, overwrite=overwrite)


def get_IWSLT(data_folder="../.data/"):
    root_path = data_folder
    extraction_filename = "2016-01.tgz"
    dataset_save_to = os.path.join(root_path, "iwslt")
    dataset_download_extract(dataset_name="iwslt", dataset_url="https://drive.google.com/uc?id=1l5y6Giag9aRPwGtuZHswh3w5v3qEz8D8")
    zpath, zroot = extract(path=root_path, filename=extraction_filename)  # ../.data/2016-01.tgz ../.data/2016-01
    print(zpath, zroot)
    process_iwslt(dataset_save_to=dataset_save_to, dataset_downloaded_in=zroot)

    # extracted folder
    print(f"Deleting folder: {zroot}")
    shutil.rmtree(zroot)

    # remove downloaded file
    extraction_file = os.path.join(root_path, extraction_filename)
    print(f"Deleting file: {extraction_file}")
    os.remove(extraction_file)


if __name__ == "__main__":
    """ downloads all  english german to <porject_folder>/.data and processes it so that torchtext.datasets.IWSLT.splits(...) works """
    get_IWSLT()
