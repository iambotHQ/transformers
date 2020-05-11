import re
import xml.etree.ElementTree as ET
from pathlib import Path
from string import punctuation
from typing import Iterable, List, Optional, Sequence, Union, TextIO, IO, List, Callable

from fire import Fire
from tqdm.auto import tqdm, trange
from logzero import logger
import multiprocessing as mp
import time

font_tags = re.compile(r"\{C:\$\w+\}")
tags_to_remove = re.compile(r"(?:(?:tłumaczenie(?: i napisy)?)|(?:korekta)|(?:synchro)|(?:hatak)|(?:przedstawia napisy)|(?:napisy)|(?:kontakt))\s*\:?")


def worker(queue: mp.JoinableQueue, batch_lines_queue: mp.JoinableQueue, processing_func: Callable[[str], List[str]]):
    while True:
        arg = queue.get()
        if int(mp.current_process().name) == 0:
            logger.info(f"There are {queue.qsize()} more files to parse")
        batch_lines_queue.put(processing_func(arg))
        queue.task_done()


def writer(path: Union[str, Path], batch_lines_queue: mp.JoinableQueue):
    time.sleep(10)
    with open(path, "w") as fhd:
        while True:
            lines = batch_lines_queue.get()
            if lines is None:
                break
            logger.warning(f"MP {mp.current_process().name} - left {batch_lines_queue.qsize()} batches, writing {len(lines)} lines")
            fhd.writelines(f"{line}\n" for line in lines)
            batch_lines_queue.task_done()


def get_texts(tree: ET.ElementTree) -> List[str]:
    return ["".join(elem.itertext()).strip() for elem in tree.findall(".//s")]


def parse_punctuation_and_remove_weird_stuff(lines: List[str]) -> List[str]:
    counter: int = 1
    while counter != len(lines):
        if tags_to_remove.search(lines[counter - 1].lower()):
            del lines[counter - 1 : counter]
            continue
        lines[counter - 1] = font_tags.sub("", lines[counter - 1])
        if lines[counter] in punctuation:
            lines[counter - 1] += lines[counter]
            del lines[counter]
        else:
            counter += 1
    return lines


def parse(path: Union[str, Path]) -> List[str]:
    tree = ET.parse(str(path))
    return parse_punctuation_and_remove_weird_stuff(get_texts(tree))


def main(root_dir: Union[str, Path], outfile: Union[str, Path]):
    paths_queue: mp.JoinableQueue = mp.JoinableQueue()
    batch_lines_queue: mp.JoinableQueue = mp.JoinableQueue()

    processes: List[mp.Process] = []
    for i in range(mp.cpu_count() - 2):
        worker_process = mp.Process(target=worker, args=(paths_queue, batch_lines_queue, parse), daemon=True, name=str(i))
        worker_process.start()
        processes.append(worker_process)

    writer_process: mp.Process = mp.Process(target=writer, args=(outfile, batch_lines_queue), daemon=True, name="Writer")
    writer_process.start()

    for path in Path(root_dir).rglob("*.xml"):
        paths_queue.put(path)

    for _ in range(len(processes)):
        paths_queue.put(None)

    for process in processes:
        process.join()
        
    batch_lines_queue.put(None)
    writer_process.join()

def fire_main():
    Fire(main)


if __name__ == "__main__":
    fire_main()
