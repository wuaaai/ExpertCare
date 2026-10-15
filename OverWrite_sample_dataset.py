import pickle
from collections import Counter
from typing import Dict, List

from pyhealth.datasets.utils import list_nested_levels, flatten_list
from torch.utils.data import Dataset

class SampleBaseDataset(Dataset):


    def __init__(self, samples: List[Dict], dataset_name="", task_name=""):
        self.samples = samples
        self.dataset_name: str = dataset_name
        self.task_name: str = task_name
        self.type_ = "base"

    def __getitem__(self, index) -> Dict:

        return self.samples[index]

    def __str__(self):

        return f"Sample dataset {self.dataset_name} {self.task_name}"

    def __len__(self):

        return len(self.samples)

    def get_all_tokens(
            self, key: str, remove_duplicates: bool = True, sort: bool = True
    ) -> List[str]:

        input_type = self.input_info[key]["type"]

        input_dim = self.input_info[key]["dim"]

        if input_type in [float, int]:
            assert input_dim == 0, f"Cannot get tokens for vector with key {key}"

        tokens = []
        for sample in self.samples:
            if input_dim == 0:

                tokens.append(sample[key])
            elif input_dim == 2:

                tokens.extend(sample[key])
            elif input_dim == 3:

                tokens.extend(flatten_list(sample[key]))
            elif input_dim == 4:

                tokens.extend(flatten_list(flatten_list(sample[key])))
            else:
                raise NotImplementedError
        if remove_duplicates:
            tokens = list(set(tokens))
        if sort:
            tokens.sort()
        return tokens

class SampleEHRDataset(SampleBaseDataset):

    def __init__(self, samples: List[Dict], code_vocs=None, dataset_name="", task_name=""):
        super().__init__(samples, dataset_name, task_name)
        self.samples = samples
        self.code_vocs = code_vocs
        self.input_info: Dict = self._validate()
        self.patient_to_index: Dict[str, List[int]] = self._index_patient()
        self.visit_to_index: Dict[str, List[int]] = self._index_visit()
        self.type_ = "ehr"

    def _validate(self) -> Dict:

        assert all(
            [isinstance(s, dict) for s in self.samples],
        ), "Each sample should be a dict"
        keys = self.samples[0].keys()

        assert all(
            [set(s.keys()) == set(keys) for s in self.samples]
        ), "All samples should have the same keys"

        assert "patient_id" in keys, "patient_id should be in the keys"
        assert "visit_id" in keys, "visit_id should be in the keys"


        input_info = {}
        for key in keys:

            levels = set([list_nested_levels(s[key]) for s in self.samples])
            assert (
                    len(levels) == 1 and len(list(levels)[0]) == 1
            ), f"Key {key} has mixed nested list levels across samples"
            level = levels.pop()[0]
            assert level in [
                0,
                1,
                2,
                3,
            ], f"Key {key} has unsupported nested list level across samples"

            if level == 0:
                flattened_values = [s[key] for s in self.samples]
            elif level == 1:
                flattened_values = [i for s in self.samples for i in s[key]]
            elif level == 2:
                flattened_values = [j for s in self.samples for i in s[key] for j in i]
            else:
                flattened_values = [
                    k for s in self.samples for i in s[key] for j in i for k in j
                ]


            types = set([type(v) for v in flattened_values])
            assert (
                    types == set([str]) or len(types.difference(set([int, float]))) == 0
            ), f"Key {key} has mixed or unsupported types ({types}) across samples"
            type_ = types.pop()

            if level == 0:

                input_info[key] = {"type": type_, "dim": 0}
            elif level == 1:

                if type_ in [float, int]:

                    lens = set([len(s[key]) for s in self.samples])
                    assert len(lens) == 1, f"Key {key} has vectors of different lengths"
                    input_info[key] = {"type": type_, "dim": 1, "len": lens.pop()}
                else:

                    input_info[key] = {"type": type_, "dim": 2}
            elif level == 2:

                if type_ in [float, int]:
                    lens = set([len(i) for s in self.samples for i in s[key]])

                    input_info[key] = {"type": type_, "dim": 2, "len": lens.pop()}
                else:

                    input_info[key] = {"type": type_, "dim": 3}
            else:

                assert type_ in [float, int, str], f"Key {key} has unsupported type across samples"

                if type_ in [float, int]:
                    lens = set([len(j) for s in self.samples for i in s[key] for j in i])
                    assert len(lens) == 1, f"Key {key} has vectors of different lengths"
                else:
                    lens = None

                input_info[key] = {"type": type_, "dim": 4, "len": lens}

        return input_info

    def _index_patient(self) -> Dict[str, List[int]]:

        patient_to_index = {}
        for idx, sample in enumerate(self.samples):
            patient_to_index.setdefault(sample["patient_id"], []).append(idx)
        return patient_to_index

    def _index_visit(self) -> Dict[str, List[int]]:

        visit_to_index = {}
        for idx, sample in enumerate(self.samples):
            visit_to_index.setdefault(sample["visit_id"], []).append(idx)
        return visit_to_index

    @property
    def available_keys(self) -> List[str]:

        keys = self.samples[0].keys()
        return list(keys)

    def get_distribution_tokens(self, key: str) -> Dict[str, int]:

        tokens = self.get_all_tokens(key, remove_duplicates=False, sort=False)
        counter = Counter(tokens)
        return counter

    def stat(self) -> str:

        lines = list()
        lines.append(f"Statistics of sample dataset:")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Task: {self.task_name}")
        lines.append(f"\t- Number of samples: {len(self)}")
        num_patients = len(set([sample["patient_id"] for sample in self.samples]))
        lines.append(f"\t- Number of patients: {num_patients}")
        num_visits = len(set([sample["visit_id"] for sample in self.samples]))
        lines.append(f"\t- Number of visits: {num_visits}")
        lines.append(
            f"\t- Number of visits per patient: {len(self) / num_patients:.4f}"
        )
        for key in self.samples[0]:
            if key in ["patient_id", "visit_id"]:
                continue
            input_type = self.input_info[key]["type"]
            input_dim = self.input_info[key]["dim"]

            if input_dim <= 1:

                num_events = [1 for sample in self.samples]
            elif input_dim == 2:

                num_events = [len(sample[key]) for sample in self.samples]
            elif input_dim == 3:

                num_events = [len(flatten_list(sample[key])) for sample in self.samples]
            else:
                raise NotImplementedError
            lines.append(f"\t- {key}:")
            lines.append(
                f"\t\t- Number of {key} per sample: "
                f"{sum(num_events) / len(num_events):.4f}"
            )
            if input_type == str or input_dim == 0:

                lines.append(
                    f"\t\t- Number of unique {key}: {len(self.get_all_tokens(key))}"
                )
                distribution = self.get_distribution_tokens(key)
                top10 = sorted(distribution.items(), key=lambda x: x[1], reverse=True)[
                        :10
                        ]
                lines.append(f"\t\t- Distribution of {key} (Top-10): {top10}")
            else:

                vector = self.samples[0][key]
                lines.append(f"\t\t- Length of {key}: {self.input_info[key]['len']}")
        print("\n".join(lines))
        return "\n".join(lines)

if __name__ == "__main__":
    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "single_vector": [1, 2, 3],
            "list_codes": ["505800458", "50580045810", "50580045811"],       
            "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
            "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],         
            "list_list_vectors": [
                [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
                [[7.7, 8.5, 9.4]],
            ],
            "label": 1,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            "single_vector": [1, 5, 8],
            "list_codes": [
                "55154191800",
                "551541928",
                "55154192800",
                "705182798",
                "70518279800",
            ],
            "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7]],
            "list_list_codes": [["A04A", "B035", "C129"], ["A07B", "A07C"]],
            "list_list_vectors": [
                [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6]],
                [[7.7, 8.4, 1.3]],
            ],
            "label": 0,
        },
    ]

    dataset = SampleEHRDataset(samples=samples)

    dataset.stat()
    data = iter(dataset)
    print(next(data))

