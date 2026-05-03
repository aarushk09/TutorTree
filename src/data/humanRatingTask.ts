export type Scenario = {
  id: string;
  studentProfile: string;
  chatContext: string;
  interventionA: string;
  interventionB: string;
};

export const scenarios: Scenario[] = [
  {
    id: "scenario_062",
    studentProfile: "Standard Beginner",
    chatContext:
      "student: How do I create a variable in Python? I tried just writing myVar = 5.\nteacher: Great start! But can you tell me what a variable actually is?\nstudent: It's just a name for a number, right?\nteacher: That's part of it, but remember, it's also about storing values. What do you think you need to do to define a variable?\nstudent: So, do I need to write something before myVar? Like 'define' or something?",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_204",
    studentProfile: "Frustrated/Overloaded Learner",
    chatContext:
      "student: Why can\u2019t I have two y-values for the same x-value?\nteacher: What do you think a function is?\nstudent: I thought it was just a rule, but now I'm confused.",
    interventionA:
      "direct_answer: The answer is the correct final result. Here is the full step-by-step solution for Understanding the concept of functions and their graphs..",
    interventionB:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
  },
  {
    id: "scenario_298",
    studentProfile: "Frustrated/Overloaded Learner",
    chatContext:
      "student: I keep thinking lists can only have numbers. Is that right?\nteacher: Lists can hold different types of data. Can you think of an example of a mixed list?\nstudent: I don't really know... I thought they were just for numbers.",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "direct_answer: The answer is the correct final result. Here is the full step-by-step solution for Using lists and their methods.",
  },
  {
    id: "scenario_303",
    studentProfile: "Deep Misconception",
    chatContext:
      "student: If it sounds convincing, it must be true, right?\nteacher: What about arguments that seem emotional but may not be logical?\nstudent: But people wouldn\u2019t say them if they weren\u2019t true!",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_372",
    studentProfile: "Deep Misconception",
    chatContext:
      "student: I wrote a function that adds two numbers, but I can't use the variable 'sum' outside of it.\nteacher: What do you think happened to 'sum' when the function finished running?\nstudent: Isn't it supposed to be available everywhere in my code?\nteacher: What does it mean to declare a variable inside a function?\nstudent: I thought it would just be a global variable.\nstudent: Why can't I use 'sum' outside? That doesn't seem fair.",
    interventionA:
      "hint: Look back at the previous rule and try applying only the next small step.",
    interventionB:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
  },
  {
    id: "scenario_054",
    studentProfile: "Standard Beginner",
    chatContext:
      "student: I can\u2019t figure out where to plot the points for this equation. I feel lost.\nteacher: Can you tell me what the coordinates represent in a graph?\nstudent: Isn\u2019t it just the x and y values? I don\u2019t see how that helps me plot them.",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_125",
    studentProfile: "Advanced Student",
    chatContext:
      "student: I can do simple loops, but nested loops confuse me.\nteacher: What happens when you use a loop inside another loop?\nstudent: I think it runs the inner loop for each iteration of the outer loop?\nteacher: Can you give an example of where that might happen?\nstudent: I\u2019m not sure how to visualize that. It feels like too many steps.",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_348",
    studentProfile: "Deep Misconception",
    chatContext:
      "student: I thought the try block would stop any errors from happening.\nteacher: What do you think happens when an error occurs in the try block?\nstudent: It just won\u2019t run if there\u2019s an error, right?\nteacher: Actually, it will run, but it will jump to the except block. Can you see how that might be useful?\nstudent: So it doesn\u2019t stop the error; it just ignores it?",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_359",
    studentProfile: "Deep Misconception",
    chatContext:
      "student: If it follows a logical structure, it must be a strong argument.\nteacher: That\u2019s an interesting point! How can we check if the conclusion is actually true?\nstudent: I guess if it sounds right, it is?",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_325",
    studentProfile: "Deep Misconception",
    chatContext:
      "student: Why can't I use the variable defined in my function outside of it?\nteacher: What do you think happens to a variable when a function finishes running?\nstudent: I thought variables inside functions are always available to the whole program.",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_256",
    studentProfile: "Frustrated/Overloaded Learner",
    chatContext:
      "student: I don't understand the difference between inductive and deductive reasoning. They sound the same.\nteacher: It's understandable to mix them up! Let's clarify each one. What do you think deductive reasoning is?\nstudent: I think it's just about making a point that feels right, like when someone is passionate.\nteacher: That's a common misconception. Deductive reasoning is based on logic and facts. Can you think of an example?\nstudent: So, if it's not just about feelings, then I don't really know any examples.",
    interventionA:
      "direct_answer: The answer is the correct final result. Here is the full step-by-step solution for Understanding deductive and inductive reasoning.",
    interventionB:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
  },
  {
    id: "scenario_461",
    studentProfile: "Passive Learner",
    chatContext:
      "student: I don't get how to solve this quadratic equation. Can I just add something to both sides?\nteacher: What do you think happens when you add or subtract from both sides of an equation?\nstudent: I think it just makes it easier to see the answer.\nteacher: What if the equation has multiple terms? How does that affect what you do?\nstudent: I guess I\u2019m not sure how to deal with those. Shouldn\u2019t I just focus on getting the number by itself?",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_334",
    studentProfile: "Deep Misconception",
    chatContext:
      "student: Isn't it okay for a function to give different results for the same number? Like f(2) could be 3 or 5.\nteacher: Functions are defined to give one output for each input. What do you think would happen if they didn't?\nstudent: But I feel like in real life, things can change!",
    interventionA:
      "direct_answer: The answer is the correct final result. Here is the full step-by-step solution for Understanding the concept of functions.",
    interventionB:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
  },
  {
    id: "scenario_321",
    studentProfile: "Deep Misconception",
    chatContext:
      "student: I can't figure out how to solve this quadratic equation. Can I just use the slope formula?\nteacher: Let's take a step back. What do you remember about the quadratic formula?\nstudent: I think it\u2019s something like y = mx + b, right?",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "direct_answer: The answer is the correct final result. Here is the full step-by-step solution for Solving quadratic equations using the quadratic formula.",
  },
  {
    id: "scenario_422",
    studentProfile: "Passive Learner",
    chatContext:
      "student: I don't get why we need to write functions. Can't we just write everything in one block?\nteacher: What do you think would happen if you had to repeat the same code multiple times?\nstudent: It would be annoying, I guess.\nteacher: And how might using a function change that situation?\nstudent: It would make it easier? But I still don\u2019t see the point.",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_455",
    studentProfile: "Passive Learner",
    chatContext:
      "student: I don\u2019t get why I need functions. Can\u2019t I just write everything in one big block of code?\nteacher: What might be the benefits of breaking your code into smaller pieces using functions?\nstudent: It seems like it could make it easier, but I still don\u2019t see why it\u2019s necessary.",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_450",
    studentProfile: "Passive Learner",
    chatContext:
      "student: I don\u2019t see how ratios work outside of cooking. Can you explain?\nteacher: What do you think a ratio represents?\nstudent: It\u2019s just a way to compare things, like ingredients.\nteacher: Can you think of other situations where comparison might be important?\nstudent: Not really. I just thought it\u2019s only for recipes.",
    interventionA:
      "hint: Look back at the previous rule and try applying only the next small step.",
    interventionB:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
  },
  {
    id: "scenario_292",
    studentProfile: "Frustrated/Overloaded Learner",
    chatContext:
      "student: I keep trying to understand loops, but they just confuse me.\nteacher: What part of loops do you find confusing? Is it the syntax or the concept?\nstudent: I think loops just repeat one thing. Why would I need them for anything else?",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "direct_answer: The answer is the correct final result. Here is the full step-by-step solution for Understanding loops in Python.",
  },
  {
    id: "scenario_435",
    studentProfile: "Passive Learner",
    chatContext:
      "student: I wrote a while loop to count numbers, but it doesn\u2019t work if the condition is false from the start.\nteacher: What do you think happens with the loop if the condition isn\u2019t met?\nstudent: I thought it would still run once, like a function.",
    interventionA:
      "hint: Look back at the previous rule and try applying only the next small step.",
    interventionB:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
  },
  {
    id: "scenario_453",
    studentProfile: "Passive Learner",
    chatContext:
      "student: When my friend said that all cats hate dogs, I just accepted it. Isn't that true?\nteacher: What could you ask to find out if that statement really holds true for all cats?\nstudent: Maybe I could ask if every cat really acts that way, but I think it's just a fact.",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_341",
    studentProfile: "Deep Misconception",
    chatContext:
      "student: I don't think I can use the quadratic formula here because I can factor it easily.\nteacher: Remember, the quadratic formula can be applied to any quadratic equation, even those that can be factored. Can you show me the equation you're trying to solve?\nstudent: It's x^2 - 5x + 6 = 0. I can just factor it as (x-2)(x-3) right?\nteacher: That's correct, but can you explain why we could also use the quadratic formula?\nstudent: I guess I don\u2019t need the formula since I can factor it.",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_266",
    studentProfile: "Frustrated/Overloaded Learner",
    chatContext:
      "student: Why do I need to back up every argument with proof? Sometimes it feels unnecessary.\nteacher: Not every argument needs physical evidence, but underlying assumptions are crucial. Let\u2019s analyze this together.\nstudent: So, assumptions are kind of like evidence? I'm confused!",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "direct_answer: The answer is the correct final result. Here is the full step-by-step solution for Evaluating arguments.",
  },
  {
    id: "scenario_134",
    studentProfile: "Advanced Student",
    chatContext:
      "student: I solved this system of equations, but it seems like there\u2019s no solution. How can that happen?\nteacher: What do you notice about the lines represented by these equations?\nstudent: They look parallel. But I thought every system should have a solution?\nteacher: What does it mean if two lines are parallel in terms of their intersection?\nstudent: I guess they don't meet. So does that mean the system is inconsistent?",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_002",
    studentProfile: "Standard Beginner",
    chatContext:
      "student: I'm trying to use a loop to print numbers from 1 to 5, but it doesn't seem to work.\nteacher: Let's check your loop syntax. Can you show me how you wrote it?\nstudent: I wrote 'for i in 1 to 5: print(i)'.\nteacher: Almost there! How do you properly define the range in Python?\nstudent: I'm not sure. Do I need to add something to fix it?",
    interventionA:
      "hint: Look back at the previous rule and try applying only the next small step.",
    interventionB:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
  },
  {
    id: "scenario_355",
    studentProfile: "Deep Misconception",
    chatContext:
      "student: I wrote a for loop, but it only prints once. Is that normal?\nteacher: Let\u2019s check how you set up your loop. What do you expect it to do?\nstudent: I thought it would just run one time and stop.",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "direct_answer: The answer is the correct final result. Here is the full step-by-step solution for Using loops effectively.",
  },
  {
    id: "scenario_042",
    studentProfile: "Standard Beginner",
    chatContext:
      "student: I wrote a loop to count from 1 to 5, but I want to print 'Hello!' each time. How do I do that?\nteacher: Great start! Can you show me the loop you wrote?\nstudent: Sure, it's like 'for i in range(1, 6):' but I don't know how to add the print.\nteacher: You're on the right track! Where do you think you should place the print statement?\nstudent: I think inside the loop? But I don't understand why it needs to be there.",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_058",
    studentProfile: "Standard Beginner",
    chatContext:
      "student: I wrote an if-statement, but it doesn\u2019t seem to do anything.\nteacher: What condition did you set? Can you share what follows the if?\nstudent: I just set it to a number, like if x = 10. Isn\u2019t that enough?",
    interventionA:
      "hint: Look back at the previous rule and try applying only the next small step.",
    interventionB:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
  },
  {
    id: "scenario_180",
    studentProfile: "Advanced Student",
    chatContext:
      "student: I can solve a^2 + b^2 = c^2 easily, but I can't figure out how to use it in a word problem.\nteacher: What steps do you think you should take to break down a word problem involving the theorem? What key pieces of information do you need to identify?\nstudent: I know I need the lengths of the sides, but translating words into numbers is where I get stuck.",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_483",
    studentProfile: "Passive Learner",
    chatContext:
      "student: I can't figure out this logic problem. If one statement is wrong, doesn't that mean the whole thing is wrong?\nteacher: How do you think the truth of a logical statement is determined?\nstudent: I don't know, I just thought that if one part is wrong then it can't be true.",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
  {
    id: "scenario_052",
    studentProfile: "Standard Beginner",
    chatContext:
      "student: I want to write a function to calculate factorial, but I just can't figure out how to do it.\nteacher: What do you think a factorial actually means? Can you describe it?\nstudent: I think it\u2019s just multiplying the number by itself a bunch of times, right?",
    interventionA:
      "socratic_prompt: That is a great question. What do you think the first step should be based on our previous rule?",
    interventionB:
      "hint: Look back at the previous rule and try applying only the next small step.",
  },
];
