---
name: Write tests for branch
description: Write tests for branch
invokable: true
---

### Context

Your goal is to write tests for the changes implemented in this branch.
To begin with get the diffs by running this command :

```
git --no-pager diff $(git merge-base main HEAD) && git --no-pager diff --no-index /dev/null $(git ls-files --others --exclude-standard)

```

Then you should recommend some tests to write.
The name of the test should describe the behavior that is tested.
For example :

- "should send api call when clicking on the button"
- "should display properly"
- "should display the tool calls when the llm sends one"
- "should return the right impacts for mistral models"

Once I have validated the tests that you want to write, implement them.

### Implementation explanation

The name of the test file should be based on the name of the file that is tested.
in this way : "file-name.test.py"

### Check that the test files are properly written

When you are done writing a test, iterate on those tests until they pass by running `pytest`
