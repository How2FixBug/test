#!/usr/bin/env python3
"""Improve ASTRA's starter corpus by reducing repeated answers and adding varied factual Q&A."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset.txt"
PAIR_RE = re.compile(
    r"<user>\s*([\s\S]*?)\s*</user>\s*"
    r"<astra>\s*([\s\S]*?)\s*</astra>"
)
THINK_RE = re.compile(r"<think>[\s\S]*?</think>")

# Original, concise examples covering different concepts and answer styles.
CURATED = [
("Why does ice float on water?", "Ice floats because it is less dense than liquid water. As water freezes, hydrogen bonds arrange its molecules into an open crystal structure that takes up more space."),
("Why is the sky blue?", "Sunlight contains many colors. Air molecules scatter shorter blue wavelengths more strongly than longer red wavelengths, so blue light reaches our eyes from across the sky."),
("Why are sunsets often red or orange?", "At sunset, sunlight travels through more atmosphere. Much of the blue light is scattered out of the direct path, leaving more red and orange light to reach your eyes."),
("What does a computer processor do?", "A processor, or CPU, executes program instructions. It performs calculations, makes logical comparisons, and coordinates work with memory and other computer components."),
("What is RAM used for?", "RAM is fast, temporary working memory. It holds data and instructions that programs are actively using, and its contents are normally lost when the computer is turned off."),
("What is the difference between RAM and storage?", "RAM holds information the computer is using right now and is usually temporary. Storage, such as an SSD, keeps files and programs even when the computer is powered off."),
("What does a graphics card do?", "A graphics processing unit (GPU) performs many calculations in parallel, especially those needed to draw images, animate scenes, and process visual data."),
("What is an operating system?", "An operating system manages a computer's hardware and provides common services for applications. Examples include macOS, Windows, Linux, Android, and iOS."),
("What happens when I open a website?", "Your browser finds the site's server using its domain name, requests the page over the internet, receives files such as HTML, CSS, and JavaScript, and uses them to display the site."),
("What is an IP address?", "An IP address is a numerical address used to identify a device or network interface so internet traffic can be routed to the right destination."),
("What is the difference between the internet and the web?", "The internet is the global network infrastructure connecting devices. The World Wide Web is one service that uses the internet to deliver linked pages and resources."),
("What does a compiler do?", "A compiler translates source code into another form, often machine code or bytecode, so a computer or runtime can execute it."),
("What is an algorithm?", "An algorithm is a finite, well-defined sequence of steps for solving a problem or completing a task."),
("What is a variable in programming?", "A variable is a named place in a program for storing a value. The program can read that value and, depending on the language, change it."),
("What is a loop in programming?", "A loop repeats a block of instructions. It usually continues while a condition is true or for each item in a collection."),
("What is a function in programming?", "A function is a reusable block of code that performs a particular task. It can accept inputs, and it may return a result."),
("What is HTML?", "HTML is the markup language used to describe the structure and meaning of web content, such as headings, paragraphs, links, and images."),
("What is CSS?", "CSS describes how web content should look, including layout, colors, spacing, typography, and responsive behavior."),
("What is JavaScript used for?", "JavaScript is a programming language commonly used to add behavior and interactivity to websites. It can also run on servers and in other environments."),
("Why do computers use binary?", "Digital circuits can reliably distinguish two states, often represented as 0 and 1. Binary represents information using combinations of those two values."),
("What is a bit?", "A bit is the smallest standard unit of digital information. It can hold one of two values, usually written as 0 or 1."),
("How many bits are in a byte?", "A byte contains 8 bits."),
("What is machine learning?", "Machine learning is a field of computing in which systems learn patterns from data and use those patterns to make predictions or decisions, rather than relying only on hand-written rules."),
("What is a neural network?", "A neural network is a machine-learning model made of connected layers of mathematical units. During training, it adjusts numerical parameters to learn patterns from examples."),
("What is a language model?", "A language model learns statistical patterns in text and uses them to estimate or generate sequences of tokens. A chatbot is often built by training and adapting a language model."),
("Why can a language model make mistakes?", "A language model generates text from patterns learned during training. It does not automatically verify every claim against reality, and limited or misleading training data can produce confident errors."),
("What is overfitting in machine learning?", "Overfitting happens when a model learns its training examples too specifically, including noise or accidental patterns, and performs poorly on new examples."),
("What is a training dataset?", "A training dataset is the collection of examples used to adjust a model's parameters. Its accuracy, variety, and relevance strongly affect what the model learns."),
("What is validation data?", "Validation data is held aside from the training updates and used to estimate how well a model performs on examples it did not train directly on."),
("What does a tokenizer do in a language model?", "A tokenizer converts text into token IDs that a model can process. Tokens may represent characters, word pieces, whole words, or other text segments."),
("What is a database?", "A database is an organized collection of information that software can store, search, update, and manage."),
("What is an API?", "An application programming interface (API) defines how one piece of software can request data or services from another using agreed rules."),
("What is cloud computing?", "Cloud computing provides computing resources, such as storage, servers, and databases, over a network instead of requiring every user to run them locally."),
("Why does a computer need cooling?", "Processors and other components produce heat while operating. Cooling moves that heat away so components can stay within safe operating temperatures."),
("What is an SSD?", "A solid-state drive stores data in flash memory and has no moving mechanical parts. It is generally much faster at accessing data than a traditional hard disk drive."),
("What is the difference between an SSD and an HDD?", "An SSD stores data in flash memory and has no moving parts. An HDD stores data magnetically on spinning disks and uses moving read/write heads."),
("What is a computer network?", "A computer network is a group of connected devices that can exchange information and share resources using communication protocols."),
("What is encryption?", "Encryption transforms readable information into an encoded form using a key, so that only someone with the appropriate key can recover the original information."),
("What is two-factor authentication?", "Two-factor authentication requires two different kinds of evidence to sign in, such as a password and a code from an authenticator app."),
("Why should software be updated?", "Updates can fix security vulnerabilities, correct bugs, improve compatibility, and add or refine features."),
("What causes thunder?", "Lightning rapidly heats the air around it. The hot air expands suddenly and creates a pressure wave that travels outward as thunder."),
("Why do we see lightning before hearing thunder?", "Light travels much faster than sound, so the flash reaches you almost immediately while the sound takes longer to arrive."),
("Why do objects fall toward Earth?", "Earth's gravity attracts objects toward its center. Near the surface, this causes unsupported objects to accelerate downward."),
("Why does the Moon have phases?", "We see different portions of the Moon's sunlit half as it orbits Earth. These changing viewing angles create the phases."),
("Why does the Moon shine?", "The Moon does not make its own visible light. It reflects sunlight toward Earth."),
("What causes ocean tides?", "Tides are caused mainly by the Moon's gravity, with the Sun also contributing. Their gravitational effects create patterns of rising and falling sea level."),
("What is photosynthesis?", "Photosynthesis is the process by which plants, algae, and some bacteria use light energy to convert carbon dioxide and water into energy-rich sugars, releasing oxygen in many cases."),
("Why do plants need sunlight?", "Plants use light energy to power photosynthesis, which helps them make sugars that provide chemical energy and building materials for growth."),
("What is the water cycle?", "The water cycle describes how water moves through the environment by processes including evaporation, condensation, precipitation, and collection or runoff."),
("What is evaporation?", "Evaporation occurs when molecules at a liquid's surface gain enough energy to enter the gas state."),
("What is condensation?", "Condensation occurs when a gas cools or loses energy and changes into a liquid, such as water droplets forming on a cold surface."),
("What is an atom?", "An atom is the basic unit of a chemical element. It has a nucleus containing protons and usually neutrons, surrounded by electrons."),
("What is a molecule?", "A molecule is a group of two or more atoms held together by chemical bonds."),
("What is the difference between an element and a compound?", "An element contains one type of atom. A compound contains atoms of two or more different elements chemically bonded in a fixed arrangement."),
("What is density?", "Density is mass divided by volume. For the same volume, a denser material contains more mass."),
("How do you calculate the area of a rectangle?", "Multiply its length by its width: area = length × width."),
("How do you calculate the area of a triangle?", "Multiply the base by the perpendicular height, then divide by two: area = ½ × base × height."),
("What is the difference between mass and weight?", "Mass measures how much matter an object contains. Weight is the force of gravity acting on that mass, so weight can change when gravity changes."),
("What is a prime number?", "A prime number is a whole number greater than 1 that has exactly two positive divisors: 1 and itself."),
("What is the average of a set of numbers?", "The arithmetic mean is found by adding all the numbers and dividing the sum by how many numbers there are."),
("What is a fraction?", "A fraction represents part of a whole or a ratio. Its numerator is the top number and its denominator is the bottom number."),
("Why do we have leap years?", "Earth takes about 365.2422 days to orbit the Sun, not exactly 365. Adding a leap day in most years divisible by four keeps the calendar aligned with the seasons; century years are exceptions unless divisible by 400."),
("Why is the ocean salty?", "Rocks release dissolved minerals as they weather. Rivers carry some of these ions to the ocean, where water evaporates but most dissolved salts remain."),
("Why do metals conduct electricity?", "Metals contain mobile electrons that can move through the material when an electric field is applied, allowing electric current to flow."),
("What is renewable energy?", "Renewable energy comes from sources that are naturally replenished on human timescales, such as sunlight, wind, flowing water, and geothermal heat."),
("What is the difference between weather and climate?", "Weather describes short-term atmospheric conditions. Climate describes long-term patterns and averages in a region, usually measured over decades."),
("Why do we need sleep?", "Sleep supports brain function, learning, memory, mood, and physical health. The body also uses sleep time for important maintenance and recovery processes."),
("What is a vaccine?", "A vaccine trains the immune system to recognize a pathogen or part of it, helping the body respond more effectively if it encounters that pathogen later."),
("What is the scientific method?", "The scientific method is a general approach to investigating questions: observe, form a testable hypothesis, collect evidence through tests or observations, analyze results, and revise explanations when needed."),
("What makes a source reliable?", "A reliable source uses evidence, identifies where its information came from, has relevant expertise or editorial standards, and distinguishes established facts from uncertainty or opinion.")
]

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()

def main() -> None:
    if not DATASET.exists():
        raise SystemExit("dataset.txt was not found. Run this script from the ASTRA project folder.")
    raw = DATASET.read_text(encoding="utf-8")
    matches = PAIR_RE.findall(raw)
    if not matches:
        raise SystemExit("No complete <user>/<astra> examples were found.")

    # Keep at most two examples for any identical answer. This retains a little
    # prompt variation without letting repeated answers dominate training.
    answer_counts: Counter[str] = Counter()
    kept: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for user, answer in matches:
        user = user.strip()
        answer = THINK_RE.sub("<think>\n</think>", answer.strip(), count=1)
        key = (normalize(user), normalize(answer))
        answer_key = normalize(THINK_RE.sub("", answer))
        if not user or not answer or key in seen_pairs or answer_counts[answer_key] >= 2:
            continue
        seen_pairs.add(key)
        answer_counts[answer_key] += 1
        kept.append((user, answer))

    added = 0
    for user, answer in CURATED:
        key = (normalize(user), normalize(answer))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        kept.append((user, "<think>\n</think>\n" + answer))
        added += 1

    preamble = raw[:re.search(r"(?m)^<user>\s*$", raw).start()].rstrip()
    sections = [preamble, "", f"<!-- ASTRA curated additions: {added} -->"]
    for user, answer in kept:
        sections.extend(["", "<user>", user, "</user>", "<astra>", answer, "</astra>"])
    DATASET.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    print(f"Removed redundant/repeated examples; retained {len(kept) - added:,} existing examples.")
    print(f"Added {added:,} varied factual Q&A examples.")
    print(f"New dataset total: {len(kept):,} examples.")
    print("Next: python3 audit_dataset.py")

if __name__ == "__main__":
    main()
