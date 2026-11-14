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


# Short, direct conversation and everyday-help examples. These vary intent and
# response style instead of repeating one answer template.
CONVERSATION = [
    ("Hi!", "Hi! What can I help you with?"),
    ("Hey there.", "Hey! What would you like to work on?"),
    ("Good morning!", "Good morning! What can I help you with today?"),
    ("Good evening!", "Good evening! How can I help?"),
    ("What's up?", "Not much. I'm ready to help with a question or project."),
    ("How are you?", "I'm doing well and ready to help. How can I help you?"),
    ("Are you there?", "Yes, I'm here. What's up?"),
    ("Can you help me?", "Absolutely. Tell me what you're working on or what's confusing."),
    ("I have a question.", "Go ahead! What's your question?"),
    ("Thanks!", "You're welcome!"),
    ("Thank you so much.", "You're very welcome. I'm glad I could help."),
    ("That helped a lot.", "I'm glad it helped!"),
    ("I don't understand.", "No problem. Which part should I explain differently?"),
    ("Can you say that more simply?", "Sure. I'll use simpler words and break it into smaller steps."),
    ("Can you give me an example?", "Of course. An example can make the idea much easier to see. What topic should I use?"),
    ("Can you be more specific?", "Yes. Tell me which part you want more detail about, and I'll focus on that."),
    ("Please keep it short.", "Got it. I'll keep the answer concise."),
    ("Explain it step by step.", "Sure. I'll break it into clear steps and explain what each one does."),
    ("Can you check my work?", "Yes. Share your work and, if you can, include the original question too."),
    ("I made a mistake.", "That's okay. Mistakes are useful clues about what to fix. What happened?"),
    ("I finished it!", "Nice work! What was the result?"),
    ("I'm stuck on a project.", "Let's narrow it down. What are you trying to make, and what part isn't working?"),
    ("Can you brainstorm with me?", "Definitely. Tell me the topic and any limits you want the ideas to follow."),
    ("I need an idea for a game.", "Try a small game built around one satisfying action, such as sorting objects, timing jumps, or combining items. Add depth only after the core action is fun."),
    ("How do I make a project easier to finish?", "Break it into small milestones. Make the simplest working version first, test it, and add features one at a time."),
    ("What should I do when code doesn't work?", "Read the first error message, find the line it points to, and check the values and syntax around that line. Change one thing at a time so you can identify the cause."),
    ("What is debugging?", "Debugging is the process of finding, understanding, and fixing problems in a program."),
    ("What does a syntax error mean?", "A syntax error means the code doesn't follow the language's grammar, so the interpreter or compiler cannot parse it."),
    ("What is a bug in software?", "A bug is a problem in a program that causes it to behave incorrectly or unexpectedly."),
    ("What is a file extension?", "A file extension is the ending of a filename, such as .txt or .png. It often indicates the file's format."),
    ("What is a folder?", "A folder is a container used to organize files and other folders."),
    ("What is a backup?", "A backup is an extra copy of important data that can be used if the original is lost, damaged, or changed by mistake."),
    ("Why should I save my work?", "Saving writes your changes to storage so they are still available after you close the program or restart the device."),
    ("What is a screenshot?", "A screenshot is an image that captures what is currently displayed on a screen."),
    ("What does offline mean?", "Offline means a device or app is not currently connected to the internet or another network."),
    ("What is a browser?", "A browser is an app used to visit websites and display web content. Examples include Safari, Firefox, Chrome, and Edge."),
    ("What is a search engine?", "A search engine helps you find information on the web by matching your search terms with pages in its index."),
    ("What is a hyperlink?", "A hyperlink is a clickable reference that takes you to another page, file, or location."),
    ("What is a username?", "A username is a name used to identify an account. It is different from a password, which helps protect access."),
    ("Why should passwords be different?", "If one service is breached, a unique password helps prevent that password from being used to access your other accounts."),
    ("What is a passphrase?", "A passphrase is a password made from several words. A long, unique passphrase can be easier to remember and difficult to guess."),
    ("What is phishing?", "Phishing is a scam in which someone pretends to be a trusted person or organization to trick people into sharing information or opening something unsafe."),
    ("What should I do with a suspicious link?", "Don't open it. Check who sent it, verify the destination through the organization's official website, and ask a trusted adult or support team if you're unsure."),
    ("What is a notification?", "A notification is a message from an app or device that alerts you about an event, update, or activity."),
    ("What does restart mean?", "Restarting means turning a device or program off and starting it again. It can clear temporary problems."),
    ("Why is my computer slow?", "Common causes include too many active apps, limited memory, low storage space, background updates, or demanding tasks. Check activity and storage before changing settings."),
    ("What is a gigabyte?", "A gigabyte is a unit of digital storage equal to about one billion bytes in decimal units. Computers also commonly use a binary-sized unit called a gibibyte."),
    ("What is open-source software?", "Open-source software makes its source code available under a license that allows people to inspect it and, under the license's terms, use, modify, or share it."),
    ("What is a video game engine?", "A game engine is a set of tools and systems for building games, often including graphics, audio, physics, input, and scene management."),
    ("What is a game loop?", "A game loop repeatedly updates the game's logic and displays the latest frame. This keeps the game responding while it runs."),
    ("What is a variable in a game?", "A game variable stores information such as a player's score, health, position, or current level."),
    ("What is a hitbox?", "A hitbox is an invisible shape used to detect interactions, such as whether two game objects have collided."),
    ("What makes a game tutorial good?", "A good tutorial teaches one idea at a time, lets the player try it immediately, and avoids explaining features before they are needed."),
    ("Can you summarize something?", "Yes. Share the text or describe the topic, and I'll pull out the main points."),
    ("Can you compare two things?", "Sure. Tell me which two things you want compared and what matters most, such as cost, speed, features, or simplicity."),
    ("What if you don't know an answer?", "I should say when I'm uncertain rather than inventing details. I can also explain what information would help verify the answer."),
    ("Can you make things up for a story?", "Yes. I can invent characters, places, events, and dialogue for fiction, while keeping invented details separate from real-world facts."),
    ("What's the difference between a fact and an opinion?", "A fact is a claim that can be checked against evidence. An opinion expresses a judgment or viewpoint."),
    ("How do I make writing clearer?", "Use direct wording, put related ideas together, and remove sentences that repeat the same point."),
    ("Why is punctuation important?", "Punctuation shows how sentences are organized and helps readers understand pauses, questions, and relationships between ideas."),
    ("What is a paragraph?", "A paragraph is a group of related sentences that develop one main idea."),
    ("What is a synonym?", "A synonym is a word with the same or a similar meaning as another word."),
    ("What is an antonym?", "An antonym is a word with the opposite meaning of another word."),
    ("What is a metaphor?", "A metaphor describes one thing as another to suggest a comparison, without using words such as 'like' or 'as'."),
    ("What is a simile?", "A simile compares two things using words such as 'like' or 'as'."),
    ("How do I plan a presentation?", "Start with the main message, organize a few supporting points, add examples or visuals where useful, and finish with a clear conclusion."),
    ("How can I prepare for a test?", "Review the main ideas, practice recalling them without looking at notes, and work through sample questions. Short, spaced study sessions are often more useful than one long cram session."),
    ("What is active recall?", "Active recall is a study method where you try to remember information before checking your notes. It helps reveal what you know and what needs more practice."),
    ("How do I prioritize tasks?", "List what needs to be done, note deadlines, and start with tasks that are both important and time-sensitive."),
    ("What is teamwork?", "Teamwork is people cooperating toward a shared goal by communicating, contributing, and coordinating their efforts."),
    ("How do I give useful feedback?", "Be specific, describe what you observed, explain its effect, and suggest a practical next step. Focus on the work rather than insulting the person."),
    ("Why do leaves change color in autumn?", "As days get shorter and temperatures change, many trees stop producing as much chlorophyll. Other pigments become visible, creating yellow, orange, and red colors."),
    ("Why do we get shadows?", "A shadow forms when an object blocks light from reaching a surface."),
    ("Why does sound echo?", "An echo happens when sound waves reflect off a surface and return to the listener after a noticeable delay."),
    ("Why do bubbles look colorful?", "Thin films of soap bend and reflect light. Light waves reflected from the film interfere with one another, producing changing colors."),
    ("Why does a compass point north?", "A compass needle aligns with Earth's magnetic field, so its north-seeking end points roughly toward geographic north."),
    ("What is a habitat?", "A habitat is the natural environment where an organism lives and finds resources such as food, water, and shelter."),
    ("What is an ecosystem?", "An ecosystem includes living organisms and the nonliving parts of their environment, along with the interactions among them."),
    ("What is biodiversity?", "Biodiversity is the variety of living things in an area or across the planet, including differences among species, genes, and ecosystems."),
    ("What is a food chain?", "A food chain shows how energy and nutrients pass from one organism to another as organisms eat and are eaten."),
    ("What causes the seasons?", "Earth's axis is tilted as Earth orbits the Sun. Each hemisphere receives different angles and lengths of daylight during the year."),
    ("What is a fossil?", "A fossil is preserved evidence of ancient life, such as bones, shells, footprints, or impressions in rock."),
    ("What is the difference between a meteor and a meteorite?", "A meteor is the streak of light made when space material enters the atmosphere. If a piece reaches the ground, it is called a meteorite."),
    ("What is a galaxy?", "A galaxy is a huge system of stars, gas, dust, and dark matter held together by gravity."),
    ("What is the Milky Way?", "The Milky Way is the galaxy that contains our Solar System."),
    ("What is a light-year?", "A light-year is a unit of distance: how far light travels in one year, about 9.46 trillion kilometers."),
    ("What is friction?", "Friction is a force that resists sliding or motion between surfaces that touch."),
    ("What is energy?", "Energy is a measurable property of a system that can be transferred or transformed. It appears in forms such as motion, heat, light, and chemical energy."),
    ("What is the difference between heat and temperature?", "Temperature measures how hot or cold something is. Heat is energy transferred because of a temperature difference."),
    ("What is a chemical reaction?", "A chemical reaction rearranges atoms in substances to form different substances with new chemical properties."),
    ("What is a cell in biology?", "A cell is the basic structural and functional unit of living organisms."),
    ("What is DNA?", "DNA is a molecule that stores hereditary information used in the growth, development, and functioning of living organisms."),
    ("What is a map scale?", "A map scale shows how a distance on the map relates to a distance in the real world."),
    ("What is latitude?", "Latitude measures how far north or south a location is from the Equator, in degrees."),
    ("What is longitude?", "Longitude measures how far east or west a location is from the Prime Meridian, in degrees."),
    ("What is a primary source?", "A primary source is evidence created during the time being studied or by someone directly involved, such as a letter, photograph, speech, or original data."),
    ("What is a secondary source?", "A secondary source explains, interprets, or analyzes information from primary sources or other evidence."),
    ("What is inflation?", "Inflation is a general increase in prices over time, which means the same amount of money buys fewer goods and services."),
    ("How many minutes are in an hour?", "There are 60 minutes in an hour."),
    ("How many seconds are in a minute?", "There are 60 seconds in a minute."),
    ("How many sides does a hexagon have?", "A hexagon has six sides."),
    ("What is a right angle?", "A right angle measures exactly 90 degrees."),
    ("What is probability?", "Probability describes how likely an event is. It ranges from 0 for impossible to 1 for certain."),
    ("What is a hypothesis?", "A hypothesis is a proposed explanation or prediction that can be tested using evidence."),
    ("What is evidence?", "Evidence is information or observations used to support, challenge, or evaluate a claim."),
    ("What is critical thinking?", "Critical thinking means examining information carefully, checking evidence, considering alternatives, and reaching a reasoned judgment."),
    ("What is misinformation?", "Misinformation is false or inaccurate information, whether or not the person sharing it intends to deceive anyone."),
    ("How can I check whether a claim is true?", "Look for reliable sources that provide evidence, compare independent reports, check the date and context, and trace the claim back to its original source when possible."),
    ("What is a prototype?", "A prototype is an early version of a product or idea used to test how it works before building the finished version."),
    ("What is iteration?", "Iteration means repeating a process, often making improvements each time based on what you learned."),
    ("What is a milestone in a project?", "A milestone is a meaningful checkpoint that marks the completion of an important stage of a project."),
    ("How do I ask a better question?", "Include the goal, the relevant context, what you have tried, and the specific part you want help with."),
    ("What information should I include in a bug report?", "Describe what you expected, what actually happened, the steps to reproduce it, and any error message. Include the device or software version when relevant."),
    ("What is a changelog?", "A changelog is a record of notable changes made to a project across its versions."),
    ("What is version control?", "Version control tracks changes to files over time so people can review earlier versions, compare edits, and collaborate."),
    ("What is Git?", "Git is a version control system that records changes to files and helps developers work on projects together."),
    ("What is a repository?", "A repository is a project folder tracked by a version control system, often containing source files and their change history."),
    ("What is a commit in Git?", "A commit is a saved snapshot of tracked changes in a Git repository, accompanied by a message describing the update."),
    ("What is responsive web design?", "Responsive web design makes a website's layout adapt to different screen sizes and devices."),
    ("What is accessibility in software?", "Accessibility means designing software so people with different abilities can use it, including people who use keyboards, screen readers, or other assistive tools."),
    ("What is a user interface?", "A user interface is the part of a system that a person interacts with, such as buttons, menus, text, and controls."),
    ("What is user experience?", "User experience describes how a person feels about and interacts with a product or service, including how easy and effective it is to use."),
    ("What makes instructions useful?", "Useful instructions state the goal, put steps in a sensible order, use clear language, and mention important warnings or requirements."),
    ("Can you quiz me?", "Yes. Tell me the subject and difficulty, and I'll ask questions one at a time."),
    ("I got the answer wrong.", "That's part of learning. Let's look at the reasoning and find the step that needs correcting."),
    ("Can you give me a hint instead of the answer?", "Absolutely. I'll give you a small clue that helps you make progress without giving away the whole solution."),
    ("Can you check this sentence for grammar?", "Yes. Send the sentence, and I'll check its grammar while keeping your intended meaning."),
    ("Can you rewrite this more clearly?", "Sure. Share the text, and I'll make it clearer without changing the main idea."),
    ("Can you make a checklist?", "Yes. Tell me what the checklist is for, and I'll turn the tasks into clear, actionable items."),
    ("Can you organize these notes?", "Sure. Share the notes, and I'll group related ideas and give them useful headings."),
    ("Can you explain a difficult topic without jargon?", "Yes. I'll use everyday language first, define any necessary technical terms, and add an example."),
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

    # Keep a little prompt variation, but prevent repeated prompts and answers
    # from dominating the training distribution.
    answer_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    kept: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for user, answer in matches:
        user = user.strip()
        answer = THINK_RE.sub("", answer.strip(), count=1).strip()
        prompt_key = normalize(user)
        answer_key = normalize(answer)
        key = (prompt_key, answer_key)
        if (
            not user or not answer or key in seen_pairs
            or answer_counts[answer_key] >= 2
            or prompt_counts[prompt_key] >= 3
        ):
            continue
        seen_pairs.add(key)
        answer_counts[answer_key] += 1
        prompt_counts[prompt_key] += 1
        kept.append((user, answer))

    added = 0
    for user, answer in CURATED + CONVERSATION:
        prompt_key = normalize(user)
        answer_key = normalize(answer)
        key = (prompt_key, answer_key)
        if key in seen_pairs or prompt_counts[prompt_key] >= 3:
            continue
        seen_pairs.add(key)
        prompt_counts[prompt_key] += 1
        answer_counts[answer_key] += 1
        kept.append((user, answer))
        added += 1

    # Remove obsolete notes that still describe the old thinking-token format.
    preamble = """ASTRA 0.1 — DATASET

FORMAT NOTES
Each example contains one user message followed by one direct ASTRA response.
There are no thinking sections. Answers should be clear, natural, accurate,
and appropriately concise. Use longer explanations when the question needs them.
Do not invent facts when uncertain.

TRAINING QUALITY NOTES
These examples teach response patterns, not fixed scripts. Respond to the
actual request, respect context, and vary wording naturally. Keep fictional
inventions distinct from factual claims.
"""
    sections = [preamble, "", f"<!-- ASTRA curated additions: {added} -->"]
    for user, answer in kept:
        sections.extend(["", "<user>", user, "</user>", "<astra>", answer, "</astra>"])
    DATASET.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    print(f"Removed redundant/repeated examples; retained {len(kept) - added:,} existing examples.")
    print(f"Added {added:,} short conversation and everyday knowledge examples.")
    print(f"New dataset total: {len(kept):,} examples.")
    print("All legacy <think> blocks were removed.")
    print("Next: python3 audit_dataset.py")

if __name__ == "__main__":
    main()
