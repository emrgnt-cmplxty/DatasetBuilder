from vllm import LLM, SamplingParams
import random

# Randomization modifiers
COURSE_TYPES = [
    "undergraduate",
    "elective",
    "specialized",
    "computer science",
    "advanced",
    "introductory",
    "high school",
    "professional development",
    "certification",
    "bootcamp",
    "seminar",
]

PROGRAMMING_PARADIGMS = [
    "object-oriented",
    "functional",
    "procedural",
    "declarative",
    "logic-based",
    "event-driven",
    "modular",
    "symbolic",
    "aspect-oriented",
    "multi-paradigm",
]

ADDITIONAL_CONTEXT = [
    ""
    " Also, mention one real-world application where this concept is critical.",
    " Additionally, discuss a potential pitfall or misconception associated with this concept.",
    " Also, highlight its benefits over the alternative approach.",
    " Also, compare this concept with its predecessor.",
    " Additionally, share its significance in modern computing.",
    " Also, provide a historical context or origin of this concept.",
    " Also, mention one real-world application where this concept is critical.",
    " Additionally, discuss a potential pitfall or misconception associated with this concept.",
    " Also, highlight its benefits over the alternative approach.",
    " Also, compare this concept with its predecessor.",
    " Additionally, share its significance in modern computing.",
    " Also, provide a historical context or origin of this concept.",
    " Besides, can you recommend any books or resources for further reading?",
    " Also, how does this concept relate to recent trends in technology?",
    " Additionally, provide a visualization or diagram explaining this concept.",
    " Also, mention any notable personalities or experts associated with this concept.",
    " In addition, discuss the scalability aspects of this concept.",
    " Also, relate this concept to a famous algorithm or software.",
]

DEPTH = [
    "in detail",
    "in simple terms",
    "in depth",
    "in a nutshell",
    "from a historical perspective",
    "from a practical perspective",
    "highlighting its pros and cons",
    "focusing on its applications",
]

PROMPT_TEMPLATES = [
    "In a {course_type} course focused on the {programming_paradigm} paradigm in Python, the topic of {concept} is crucial. Please define a different but related concept {depth} and illustrate it with Python code.",
    "In a {course_type} course focused on the {programming_paradigm} paradigm in Python, the topic of {concept} is crucial. Please define this concept {depth} and illustrate it with Python code.",
    "There are {course_type} courses touching upon {programming_paradigm} paradigm which discuss {concept}. What is it, {depth}? Also, provide a relevant Python example.",
    "Often {course_type} courses touch upon {programming_paradigm} paradigm in Python often discuss {concept}. Can you think of a related concept and explain it {depth}? Also, provide a relevant Python example.",
    "In modern {course_type} courses, the emphasis on the {programming_paradigm} paradigm in Python has grown. Explore the topic of {concept} {depth} and provide practical Python examples.",
    "With a focus on {programming_paradigm} paradigm, how would you introduce the topic of {concept} in a {course_type} course? Break it down {depth}.",
]

CONCEPTS = [
    # Basic Mathematics:
    "numerical properties: associativity, commutativity, distributivity",
    "fundamental theorem of arithmetic",
    "prime factorization",
    "GCD and LCM relationships",
    "exponential properties",
    "logarithm properties and bases",
    "complex numbers: basic operations",
    "complex numbers: polar form",
    "fraction operations: addition, subtraction, multiplication, division",
    "decimal to fraction conversions",
    "integer and fractional exponents",
    "equation properties: transitive, reflexive, symmetric",
    # Intermediate Mathematics:
    "functions: domain and range",
    "functions: injective, surjective, bijective",
    "functions: inverse and compositions",
    "coordinate geometry: distance formula",
    "coordinate geometry: midpoint formula",
    "coordinate geometry: slope and equation of lines",
    "sequences and series: arithmetic progressions",
    "sequences and series: geometric progressions",
    "matrices: determinant and inverse",
    "matrices: rank and eigenvalues",
    "vector spaces and subspaces",
    "Basis and dimension of vector spaces",
    "linear transformations and their matrix representations",
    "Laplace and Fourier transforms",
    # Python-specific Mathematical Applications:
    "Python libraries: SymPy for symbolic mathematics",
    "Python libraries: math for basic mathematical functions",
    "Python libraries: cmath for complex numbers",
    "Python applications: solving linear equations with NumPy",
    "Python applications: plotting graphs with matplotlib",
    "Python applications: solving ordinary differential equations with SciPy",
    "Python applications: symbolic integration and differentiation with SymPy",
    "Python applications: matrix operations with NumPy",
    "Python applications: statistical analysis with pandas and NumPy",
    "Python applications: regression models with scikit-learn",
    "Python applications: optimization problems with SciPy's optimize",
    "Python applications: Fourier transformations with SciPy",
    "Python applications: generating and analyzing sequences with itertools",
    "Python applications: number theory operations with gmpy2",
    "Python applications: combinatorics with itertools and SymPy",
    "Python applications: geometry computations using libraries like Shapely",
    "Python applications: solving polynomial equations with NumPy's roots function",
    "Python applications: numerical integration using SciPy's quad function",
    "Python applications: eigenvalue and eigenvector computations with NumPy",
    "Python applications: probability distributions with SciPy.stats",
    # Additional Python Concepts:
    "Python data science: introduction to pandas DataFrame",
    "Python data science: data manipulation with pandas",
    "Python data science: visualization with seaborn",
    "Python machine learning: introduction to scikit-learn",
    "Python machine learning: clustering with KMeans",
    "Python machine learning: classification with Random Forests",
    "Python machine learning: regression with linear regression",
    "Python deep learning: introduction to TensorFlow and Keras",
    "Python deep learning: building a neural network",
    "Python deep learning: training and evaluation",
    "Python applications: web scraping with BeautifulSoup and Scrapy",
    "Python applications: web development with Flask and Django",
    "Python applications: game development with Pygame",
    "Python applications: image processing with PIL and OpenCV",
    "Python applications: desktop applications with PyQt",
    # Introductory Algebra/Mathematics/Statistics:
    "addition and subtraction",
    "multiplication and division",
    "basic statistics: mean, median, mode",
    "advanced statistics: standard deviation, variance, skewness, kurtosis",
    "combinatorial problems: permutations",
    "combinatorial problems: combinations",
    "combinatorial problems: factorials",
    "basic probability theory",
    "probability simulations",
    "probability distributions: binomial",
    "probability distributions: Poisson",
    "probability distributions: normal",
    "graphing: linear equations",
    "graphing: inequalities",
    "graphing: quadratic functions (parabolas)",
    "geometry basics: perimeter",
    "geometry basics: area",
    "geometry basics: volume",
    "trigonometric functions",
    "Taylor series",
    "polynomial approximations",
    "polynomial operations: addition",
    "polynomial operations: subtraction",
    "polynomial operations: multiplication",
    "set theory: union",
    "set theory: intersection",
    "set theory: difference",
    "search algorithms: linear",
    "search algorithms: binary",
    "regression analysis",
    "time series analysis",
    "forecasting",
    "confidence intervals",
    "hypothesis testing",
    # Extended Introductory Algebra/Mathematics/Statistics:
    "advanced algebra: matrix operations",
    "advanced algebra: eigenvalues and eigenvectors",
    "calculus: differentiation",
    "calculus: integration",
    "calculus: partial differentiation",
    "calculus: multivariable integration",
    "calculus: fundamental theorem",
    "calculus: chain rule",
    "calculus: product and quotient rules",
    "number theory: prime numbers",
    "number theory: greatest common divisor",
    "number theory: least common multiple",
    "number theory: modulo arithmetic",
    "discrete math: graphs",
    "discrete math: trees",
    "discrete math: logic gates",
    "differential equations: first order",
    "differential equations: second order",
    "linear algebra: vectors",
    "linear algebra: dot product and cross product",
    "linear algebra: systems of equations",
    "optimization problems: linear programming",
    "optimization problems: simplex method",
    # Basic Python concepts:
    "Python syntax: for loops",
    "Python syntax: conditional statements",
    "Python syntax: list comprehensions",
    "Python data structures: stacks",
    "Python data structures: queues",
    "Python data structures: lists",
    "Python data structures: tuples",
    "Python data structures: dictionaries",
    "Python data structures: sets",
    "Python basics: unpacking",
    "Python basics: lambda functions",
    "Python basics: ternary conditional expressions",
    "Python basics: string operations",
    "Python file operations: reading",
    "Python file operations: writing",
    "Python file operations: exception handling",
    "Python practices: PEP8",
    "Python practices: naming conventions (camelCase, snake_case)",
    "Python modules: datetime",
    "Python modules: time",
    "Python modules: collections",
    "Python modules: itertools",
    "Python memory concepts: garbage collection",
    "Python memory concepts: reference cycles",
    "Python memory concepts: deep copy vs shallow copy",
    "Python introspection: decorators",
    "Python introspection: higher-order functions",
    "Python introspection: introspection techniques",
    "Python introspection: reflection techniques",
    "Python OOP: setters",
    "Python OOP: getters",
    "Python OOP: inheritance",
    "Python OOP: composition",
    "Python OOP: encapsulation",
    "Python OOP: polymorphism",
    "External Python libraries: NumPy",
    "External Python libraries: pandas",
    "External Python libraries: matplotlib",
    "Advanced Python topics: recursion",
    "Advanced Python topics: iteration",
    "Advanced Python topics: bitwise operations",
    "Advanced Python topics: serialization (JSON, Pickle)",
    # Extended Basic Python concepts:
    "Python exceptions: custom exceptions",
    "Python exceptions: raising exceptions",
    "Python functions: nested functions",
    "Python functions: closures",
    "Python functions: decorators",
    "Python functions: function factories",
    "Python metaclasses: basics",
    "Python metaclasses: use cases",
    "Python concurrency: threading",
    "Python concurrency: multiprocessing",
    "Python concurrency: async/await",
    "Python concurrency: coroutines",
    "Python networking: socket programming",
    "Python networking: async networking",
    "Python GUI: Tkinter basics",
    "Python GUI: PyQt basics",
    "Python GUI: wxPython basics",
    "Python database: SQLite integration",
    "Python database: SQLAlchemy basics",
    "Python advanced topics: magic methods (dunder methods)",
    # Core Computer Science Principles:
    "CS basics: static typing",
    "CS basics: dynamic typing",
    "CS basics: public access modifiers",
    "CS basics: private access modifiers",
    "CS basics: strong typing",
    "CS basics: weak typing",
    "Programming paradigms: procedural",
    "Programming paradigms: functional",
    "Programming paradigms: OOP",
    "Programming paradigms: event-driven",
    "Programming paradigms: reactive",
    "Programming paradigms: declarative",
    "Programming paradigms: imperative",
    "Concurrency: multi-threading",
    "Concurrency: synchronicity",
    "Concurrency: thread safety",
    "Concurrency: race conditions",
    "Memory concepts: Memory layout",
    "Memory concepts: garbage collection",
    "Memory concepts: reference counting",
    "Graph theory: DFS",
    "Graph theory: BFS",
    "Graph theory: cycles",
    "Graph theory: trees",
    "Graph theory: forests",
    "Graph theory: traversal techniques",
    "Graph theory: topological sorting",
    "Algorithms & design: greedy algorithms",
    "Algorithms & design: dynamic programming",
    "Algorithms & design: backtracking",
    "Algorithms & design: BFS",
    "Algorithms & design: DFS",
    "Algorithms & design: heuristics",
    "Computer architecture: static libraries",
    "Computer architecture: dynamic libraries",
    "Computer architecture: just-in-time compilation",
    "Computer architecture: ahead-of-time compilation",
    "Networking & web: RESTful web services",
    "Networking & web: client-side scripting",
    "Networking & web: server-side scripting",
    "Networking & web: microservices architecture",
    "Networking & web: monolithic architecture",
    "Databases: transactional",
    "Databases: non-transactional",
    "Databases: relational",
    "Databases: non-relational",
    "Databases: SQL systems",
    "Databases: NoSQL systems",
    "Databases: inner joins",
    "Databases: outer joins",
    "Data structures: continuous",
    "Data structures: discrete",
    "Data structures: binary trees",
    "Data structures: balanced search trees",
    "Data structures: linked lists",
    "Data structures: arrays",
    "Data structures: hash tables",
    "Patterns & designs: MVC",
    "Patterns & designs: Singleton",
    "Patterns & designs: Factory",
    "Patterns & designs: Observer",
    "Patterns & designs: Event-driven",
    "Patterns & designs: refactoring",
    "Patterns & designs: code smells",
    "Languages & scripting: interpretive languages",
    "Languages & scripting: compiled languages",
    "Languages & scripting: deterministic algorithms",
    "Languages & scripting: nondeterministic algorithms",
    "Functional programming: closures",
    "Functional programming: first-class functions",
    "Functional programming: higher-order functions",
    "Functional programming: tail call optimization",
    "Security: authentication",
    "Security: authorization",
    "Security: encryption",
    "Security: secure programming practices",
    "Development concepts: unit testing",
    "Development concepts: integration testing",
    # Extended Algorithms & Design:
    "Divide and conquer algorithms",
    "Merge sort",
    "Quick sort",
    "Radix sort",
    "Kruskal’s algorithm for MST",
    "Prim’s algorithm for MST",
    "Dijkstra’s shortest path algorithm",
    "Floyd-Warshall all-pairs shortest path algorithm",
    "Bellman-Ford shortest path algorithm",
    "Knapsack problem",
    "Job sequencing with deadlines",
    "Huffman coding",
    "Union-Find algorithms",
    "KMP string matching algorithm",
    "Rabin-Karp string matching algorithm",
    "Z algorithm for string matching",
    "Minimum window substring problem",
    "Algorithmic paradigms: sliding window",
    "Algorithmic paradigms: two pointers",
    "Algorithmic paradigms: meet in the middle",
    "Amortized analysis",
    "Max flow and min cut algorithms",
    "Topological sort (Kahn's and DFS-based)",
    "Red-Black trees",
    "AVL trees",
    "Segment trees",
    "Trie data structure",
    "Suffix tree and Suffix array",
    "NP-completeness and reductions",
    "Approximation algorithms",
]

# Instruction template
INSTRUCTION_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def main(
    model="WizardLM/WizardCoder-Python-13B-V1.0",
    batch_size=1_024,
    temperature=1,
    top_p=0.9,
    top_k=40,
    max_tokens=1_024,
):
    """Run the textbook generator."""

    instructions, results = [], []

    model_config = {
        "model": model,
        "batch_size": batch_size,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
    }

    for _ in range(batch_size):
        # Random selections
        prompt_template = random.choice(PROMPT_TEMPLATES)
        course_type = random.choice(COURSE_TYPES)
        programming_paradigm = random.choice(PROGRAMMING_PARADIGMS)
        concept = random.choice(CONCEPTS)
        depth = random.choice(DEPTH)
        additional_context = random.choice(ADDITIONAL_CONTEXT)

        # Assembling the prompt:
        prompt = prompt_template.format(
            course_type=course_type,
            programming_paradigm=programming_paradigm,
            concept=concept,
            depth=depth,
            additional_context=additional_context,
        )

        config = {
            "prompt_template": prompt_template,
            "course_type": course_type,
            "programming_paradigm": programming_paradigm,
            "concept": concept,
            "depth": depth,
            "additional_context": additional_context,
            **model_config,
        }

        instructions.append(INSTRUCTION_TEMPLATE.format(instruction=prompt))
        results.append(config)

    llm = LLM(model=model)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(instructions, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("-" * 100)
        print("Prompt: ", prompt)
        print("Generated Text: ", generated_text)
        print("-" * 100)
        results[i]["generated_text"] = generated_text

    print("Final Results = ", results)


if __name__ == "__main__":
    import fire

    fire.Fire({"cli": main})
