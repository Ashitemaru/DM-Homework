from typing import Dict, Set, List
from tqdm import tqdm

config = {
    "dataset_path": "./Groceries_dataset.csv",
    "support_threshold": 30,
}

class DataSet:
    def __init__(self, file_path: str) -> None:
        self.data: Dict[Set[str]] = {}
        self.item_set: Set[str] = set()
        self.tot_cnt = 0

        handle = open(file_path, "r")
        raw_data = handle.readlines()[1: ] # Pass the head of the table
        handle.close()

        for data_line in raw_data:
            x = data_line.strip().split(",")
            key = x[0] + " @ " + x[1]
            if key not in self.data:
                self.data[key] = set()

            self.data[key].add(x[2])
            self.item_set.add(x[2])

class FreqSet:
    def __init__(self, data_set: DataSet) -> None:
        self.k = 1
        self.freq_sets: List[Dict[Set[str]]] = [{}]
        self.data_set = data_set

        for tx_key, item_set in data_set.data.items():
            for item in item_set:
                if item not in self.freq_sets[0]:
                    self.freq_sets[0][item] = set()
                self.freq_sets[0][item].add(tx_key)

    def step(self) -> None:
        self.k += 1
        print("Step k from %d to %d." % (self.k - 1, self.k))

        def calc_support_set(item_list: List[str]) -> Set[str]:
            res = set()
            for tx_key, item_set in self.data_set.data.items():
                all_contained = True
                for item in item_list:
                    if item not in item_set:
                        all_contained = False
                        break
                if all_contained:
                    res.add(tx_key)
            return res

        def concat(item_list: List[str]) -> str:
            tmp = item_list.copy()
            tmp.sort()
            return " & ".join(tmp)

        self.freq_sets.append({})
        if self.k == 2:
            for item_a in tqdm(self.data_set.item_set):
                for item_b in self.data_set.item_set:
                    if item_a >= item_b:
                        continue
                    support_set = calc_support_set([item_a, item_b])
                    if len(support_set) > 0:
                        self.freq_sets[-1][item_a + " & " + item_b] = support_set
        elif self.k == 3:
            base = [(k, len(v)) for k, v in self.freq_sets[-2].items()]
            base.sort(key = lambda x: -x[1])
            base = base[: config["support_threshold"]]

            key_set = set()
            for item_c in tqdm(self.data_set.item_set):
                for item_ab in base:
                    item_a, item_b = item_ab[0].split(" & ")

                    if item_c == item_a or item_c == item_b:
                        continue

                    new_key = concat([item_a, item_b, item_c])
                    if new_key in key_set:
                        continue
                    else:
                        key_set.add(new_key)
                    
                    support_set = calc_support_set([item_a, item_b, item_c])
                    if len(support_set) > 0:
                        self.freq_sets[-1][new_key] = support_set
        else:
            raise NotImplementedError()

    def print_most_freq(self, k: int, range: int) -> None:
        base = [(k, len(v)) for k, v in self.freq_sets[k - 1].items()]
        base.sort(key = lambda x: -x[1])
        for ind, pair in enumerate(base[: range]):
            print("No.%d, Key: %s, Count: %d, Support: %.6f" %
                (ind, pair[0], len(self.freq_sets[k - 1][pair[0]]),
                 len(self.freq_sets[k - 1][pair[0]]) / len(self.data_set.data)))

            # Print confidence when k == 2
            if k == 2:
                item_pair = pair[0].split(" & ")
                p_xy = len(self.freq_sets[1][pair[0]])
                print("Confidence %s -> %s: %.6f, %s -> %s: %.6f" %
                    (item_pair[0], item_pair[1], p_xy / len(self.freq_sets[0][item_pair[0]]),
                     item_pair[1], item_pair[0], p_xy / len(self.freq_sets[0][item_pair[1]])))

    def print_freq_set(self) -> None:
        for k, v in self.freq_sets[self.k - 1].items():
            print("Key: %s, Value: %s" % (k, str(v)))

def main() -> None:
    data_set = DataSet(config["dataset_path"])
    freq_set = FreqSet(data_set)

    # k == 1
    freq_set.print_most_freq(1, 10)

    # k == 2
    freq_set.step()
    freq_set.print_most_freq(2, 5)

    # k == 3
    freq_set.step()
    freq_set.print_most_freq(3, 5)

if __name__ == "__main__":
    main()