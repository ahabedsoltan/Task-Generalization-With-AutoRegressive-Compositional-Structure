import numpy as np
import torch
from torch.utils.data import Dataset
import tqdm
from .my_dataset import MyDataset
import tiktoken
import random
from itertools import permutations, product

class SyntheticMultiLangRandDataset(MyDataset):
    """
    Generates synthetic data for multiple languages based on a specified number of languages.

    For each sample:
    - Lines 1..(n_lines-1):
        [meaning i in lang1], [meaning i in lang2], ..., [meaning i in langN]
    - Line n_lines:
        [meaning n in lang1], ([meaning n in lang2], ..., [meaning n in langN])

    - The word on the same line (meaning i) has the same semantic meaning across languages.
    - The model sees lines 1..(n_lines-1) fully and line n_lines partially (only the first token).
      It must predict the remaining tokens in the final line.
    """

    # Define a fixed ordered list of 10 languages
    AVAILABLE_LANGUAGES = [
        "English", "French", "Spanish", "Chinese",
        "German", "Italian", "Japanese", "Russian",
        "Portuguese", "Arabic"
    ]

    def __init__(
        self,
        kwargs,
        generate=True
    ):
        """
        :param kwargs: Dictionary containing the following keys:
            - split: "train" or "test"
            - n_samples: number of examples to generate
            - n_lines: how many lines per example (each line has tokens in different languages)
            - output_dir: where to save the dataset
            - seed: random seed for reproducibility
            - num_languages: number of languages to include (integer, max 10)
            - n_steps: number of languages per line (must be <= num_languages)
            - n_tasks: number of language combinations per split
        :param generate: Whether to generate the dataset immediately
        """
        super().__init__(kwargs, generate)
        np.random.seed(kwargs['seed'])
        random.seed(kwargs['seed'])
        self.split = kwargs['split']
        self.n_samples = kwargs['n_samples']
        self.n_lines = kwargs['n_lines']
        self.n_steps = kwargs['n_steps']
        self.n_tasks = kwargs['n_tasks']
        self.num_languages = kwargs['num_languages']  # Number of languages to include

        # Validate num_languages
        if not (1 <= self.num_languages <= len(self.AVAILABLE_LANGUAGES)):
            raise ValueError(f"num_languages must be between 1 and {len(self.AVAILABLE_LANGUAGES)}")

        # Select the first `num_languages` from the available languages
        self.languages = self.AVAILABLE_LANGUAGES[:self.num_languages]

        # # Validate n_steps
        # if self.n_steps > self.num_languages:
        #     raise ValueError(f"n_steps ({self.n_steps}) cannot exceed num_languages ({self.num_languages})")

        # Initialize translations for all available languages
        self.meaning2translation = self._initialize_meanings(self.AVAILABLE_LANGUAGES)

        # Define the test triplet based on the selected languages
        self.test_triplet = tuple(self.languages)  # e.g., ("English", "French", "Spanish")

        self.input_ids = []
        self.labels = []
        self.attention_mask = []

        # Initialize the tokenizer (tiktoken GPT2). Replace with any other if desired.
        self.encoder = tiktoken.get_encoding("gpt2")
        
        
        print('here')
        # Generate data
        if self.do_generate:
            self._generate_dataset()
        
    def _initialize_meanings(self, languages):
        """
        Initialize the meanings with translations for all specified languages.

        :param languages: List of language names
        :return: List of dictionaries with translations
        """
        # List of meanings
        base_meanings = [
            "cat", "dog", "house", "apple", "sky",
            "car", "road", "tree", "bed", "water",
            "sun", "moon", "star", "book", "phone",
            "computer", "flower", "chair", "table", "mountain"
        ]

        # Actual translations for each meaning across 10 languages
        meaning2translation = [
            {
                "English": "cat",
                "French": "chat",
                "Spanish": "gato",
                "Chinese": "猫",          # māo
                "German": "Katze",
                "Italian": "gatto",
                "Japanese": "猫",         # neko
                "Russian": "кот",
                "Portuguese": "gato",
                "Arabic": "قط",           # qiṭṭ
            },
            {
                "English": "dog",
                "French": "chien",
                "Spanish": "perro",
                "Chinese": "狗",          # gǒu
                "German": "Hund",
                "Italian": "cane",
                "Japanese": "犬",         # inu
                "Russian": "собака",
                "Portuguese": "cão",
                "Arabic": "كلب",          # kalb
            },
            {
                "English": "house",
                "French": "maison",
                "Spanish": "casa",
                "Chinese": "房子",         # fángzi
                "German": "Haus",
                "Italian": "casa",
                "Japanese": "家",          # ie
                "Russian": "дом",
                "Portuguese": "casa",
                "Arabic": "منزل",         # manzil
            },
            {
                "English": "apple",
                "French": "pomme",
                "Spanish": "manzana",
                "Chinese": "苹果",         # píngguǒ
                "German": "Apfel",
                "Italian": "mela",
                "Japanese": "りんご",      # ringo
                "Russian": "яблоко",
                "Portuguese": "maçã",
                "Arabic": "تفاح",          # tuffāḥ
            },
            {
                "English": "sky",
                "French": "ciel",
                "Spanish": "cielo",
                "Chinese": "天空",         # tiānkōng
                "German": "Himmel",
                "Italian": "cielo",
                "Japanese": "空",          # sora
                "Russian": "небо",
                "Portuguese": "céu",
                "Arabic": "سماء",          # samā’
            },
            {
                "English": "car",
                "French": "voiture",
                "Spanish": "coche",
                "Chinese": "车",           # chē
                "German": "Auto",
                "Italian": "macchina",
                "Japanese": "車",          # kuruma
                "Russian": "машина",
                "Portuguese": "carro",
                "Arabic": "سيارة",         # sayyāra
            },
            {
                "English": "road",
                "French": "route",
                "Spanish": "carretera",
                "Chinese": "路",           # lù
                "German": "Straße",
                "Italian": "strada",
                "Japanese": "道",          # michi
                "Russian": "дорога",
                "Portuguese": "estrada",
                "Arabic": "طريق",          # ṭarīq
            },
            {
                "English": "tree",
                "French": "arbre",
                "Spanish": "árbol",
                "Chinese": "树",           # shù
                "German": "Baum",
                "Italian": "albero",
                "Japanese": "木",          # ki
                "Russian": "дерево",
                "Portuguese": "árvore",
                "Arabic": "شجرة",          # shajarah
            },
            {
                "English": "bed",
                "French": "lit",
                "Spanish": "cama",
                "Chinese": "床",           # chuáng
                "German": "Bett",
                "Italian": "letto",
                "Japanese": "ベッド",       # beddo
                "Russian": "кровать",
                "Portuguese": "cama",
                "Arabic": "سرير",          # sarīr
            },
            {
                "English": "water",
                "French": "eau",
                "Spanish": "agua",
                "Chinese": "水",           # shuǐ
                "German": "Wasser",
                "Italian": "acqua",
                "Japanese": "水",          # mizu
                "Russian": "вода",
                "Portuguese": "água",
                "Arabic": "ماء",           # mā’
            },
            {
                "English": "sun",
                "French": "soleil",
                "Spanish": "sol",
                "Chinese": "太阳",         # tàiyáng
                "German": "Sonne",
                "Italian": "sole",
                "Japanese": "太陽",         # taiyō
                "Russian": "солнце",
                "Portuguese": "sol",
                "Arabic": "شمس",           # shams
            },
            {
                "English": "moon",
                "French": "lune",
                "Spanish": "luna",
                "Chinese": "月亮",         # yuèliang
                "German": "Mond",
                "Italian": "luna",
                "Japanese": "月",          # tsuki
                "Russian": "луна",
                "Portuguese": "lua",
                "Arabic": "قمر",           # qamar
            },
            {
                "English": "star",
                "French": "étoile",
                "Spanish": "estrella",
                "Chinese": "星星",         # xīngxing
                "German": "Stern",
                "Italian": "stella",
                "Japanese": "星",          # hoshi
                "Russian": "звезда",
                "Portuguese": "estrela",
                "Arabic": "نجم",           # najm
            },
            {
                "English": "book",
                "French": "livre",
                "Spanish": "libro",
                "Chinese": "书",           # shū
                "German": "Buch",
                "Italian": "libro",
                "Japanese": "本",          # hon
                "Russian": "книга",
                "Portuguese": "livro",
                "Arabic": "كتاب",          # kitāb
            },
            {
                "English": "phone",
                "French": "téléphone",
                "Spanish": "teléfono",
                "Chinese": "电话",         # diànhuà
                "German": "Telefon",
                "Italian": "telefono",
                "Japanese": "電話",         # denwa
                "Russian": "телефон",
                "Portuguese": "telefone",
                "Arabic": "هاتف",          # hātif
            },
            {
                "English": "computer",
                "French": "ordinateur",
                "Spanish": "computadora",
                "Chinese": "电脑",         # diànnǎo
                "German": "Computer",
                "Italian": "computer",
                "Japanese": "コンピュータ",   # konpyūta
                "Russian": "компьютер",
                "Portuguese": "computador",
                "Arabic": "حاسوب",          # ḥāsūb
            },
            {
                "English": "flower",
                "French": "fleur",
                "Spanish": "flor",
                "Chinese": "花",           # huā
                "German": "Blume",
                "Italian": "fiore",
                "Japanese": "花",          # hana
                "Russian": "цветок",
                "Portuguese": "flor",
                "Arabic": "زهرة",           # zahrat
            },
            {
                "English": "chair",
                "French": "chaise",
                "Spanish": "silla",
                "Chinese": "椅子",         # yǐzi
                "German": "Stuhl",
                "Italian": "sedia",
                "Japanese": "椅子",         # isu
                "Russian": "стул",
                "Portuguese": "cadeira",
                "Arabic": "كرسي",           # kursī
            },
            {
                "English": "table",
                "French": "table",
                "Spanish": "mesa",
                "Chinese": "桌子",         # zhuōzi
                "German": "Tisch",
                "Italian": "tavolo",
                "Japanese": "テーブル",     # tēburu
                "Russian": "стол",
                "Portuguese": "mesa",
                "Arabic": "طاولة",          # ṭāwila
            },
            {
                "English": "mountain",
                "French": "montagne",
                "Spanish": "montaña",
                "Chinese": "山",           # shān
                "German": "Berg",
                "Italian": "montagna",
                "Japanese": "山",          # yama
                "Russian": "гора",
                "Portuguese": "montanha",
                "Arabic": "جبل",           # jabal
            }
        ]

        # Verify that each meaning has translations for all available languages
        for meaning in meaning2translation:
            for lang in languages:
                if lang not in meaning:
                    raise ValueError(f"Missing translation for language '{lang}' in meaning '{meaning['English']}'")

        return meaning2translation

    def _generate(self):
        pass

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

    def _generate_dataset(self):
        lang_triplets = self._get_lang_triplets()
        num_triplets = len(lang_triplets)
        print(f"Number of language combinations: {num_triplets}")
        if num_triplets == 0:
            print("No triplets available for this split!")
            return

        # Distribute the total samples among the possible triplets
        samples_per_triplet = max(1, self.n_samples // num_triplets)
        samples = 0

        for trip in tqdm.tqdm(lang_triplets, desc="Generating data"):
            for _ in range(samples_per_triplet):
                if samples >= self.n_samples:
                    break
                samples += 1
                self._generate_example_for_triplet(trip)
            if samples >= self.n_samples:
                break
        self.padding()

    def padding(self):
        pad_token_id = 0  # GPT-2 pad token
        length = max(len(tokens) for tokens in self.input_ids)
        new_input_ids = []
        new_attention_mask = []
        new_labels = []
        for input_id, attention, label in zip(self.input_ids, self.attention_mask, self.labels):
            pad_length = length - len(input_id)
            new_input_ids.append(input_id + [pad_token_id] * pad_length)
            new_labels.append(label + [-100] * pad_length)
            new_attention_mask.append(attention + [0] * pad_length)
        self.input_ids = new_input_ids
        self.attention_mask = new_attention_mask
        self.labels = new_labels

    def _get_lang_triplets(self):
        """
        Returns the set of language combinations for the given split:
        - train => all permutations except the test_triplet
        - test  => only the test_triplet
        """
        all_combos = list(product(self.languages, repeat = self.n_steps))  

        random.shuffle(all_combos)

        if self.split == "test":
            # Only the specified test_triplet in test
            return all_combos[self.n_tasks:]
        else:
            # Exclude the test_triplet from training
            return all_combos[:self.n_tasks]

    def _generate_example_for_triplet(self, triplet):
        """
        Build a single example with n_lines lines, each line using a single shared meaning across languages.
        For lines 1..(n_lines-1), we reveal all tokens.
        For line n_lines, we only show the first token; the model must predict the remaining tokens.
        """
        lines = []

        for i in range(1, self.n_lines + 1):
            # Randomly pick a meaning from the dictionary
            meaning_dict = random.choice(self.meaning2translation)

            words = [lang + ' ' + meaning_dict[lang] for lang in triplet]

            if i < self.n_lines:
                # Fully reveal all tokens
                line = " , ".join(words)
            else:
                # Final line: only reveal the first token; others are in parentheses
                revealed = words[0]
                hidden = " , ".join(words[1:])
                line = f"{revealed} , ( {hidden} )"

            lines.append(line)

        # Combine lines into a single string example
        example_text = "\n".join(lines)

        #        # Combine lines into a single string example
        example_text = "\n".join(lines)

        # Tokenize once for the entire example
        tokens = self.encoder.encode(example_text)

        # Create label array, defaulting to -100
        labels = [-100] * len(tokens)

        # We need to identify the tokens corresponding to the final line's second & third words
        line_token_list = []
        offset = 0
        for line in lines:
            line_toks = self.encoder.encode(line)
            line_token_list.append((offset, offset + len(line_toks)))
            offset += len(line_toks)

        # The last line offsets
        last_line_start, last_line_end = line_token_list[-1]
        last_line = lines[-1]

        # Split out chunk1 (the revealed word) vs chunk2 (the parentheses with the hidden words)
        splitted = last_line.split("(", 1)
        chunk1 = splitted[0]  # e.g. "cat "
        chunk2 = "(" + splitted[1]  # e.g. "(chien perro)" or similar
        chunk2_toks = self.encoder.encode(chunk2)
        # Assign the correct token IDs to labels in the parentheses
        for pos in range(1, len(chunk2_toks)):
            labels[-pos] = tokens[-pos]

        attention_mask = [1] * len(tokens)

        # Append to our dataset
        self.input_ids.append(tokens)
        self.labels.append(labels)
        self.attention_mask.append(attention_mask)

    def save(self, output_dir):
        super().save(output_dir)


    def get_name(self):
        """
        Generate a unique name for the dataset based on its configuration.
        """
        return f"synthetic_multilang_rand2_{self.split}_{self.num_languages}_{self.n_samples}_{self.n_lines}_{self.n_tasks}_{self.n_steps}"


