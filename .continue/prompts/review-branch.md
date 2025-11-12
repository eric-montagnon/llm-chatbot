---
name: Review branch
description: Review branch
invokable: true
---

### Context

Your goal is to detect possible improvements in the changes between this branch and the main branch it was originally branched off

To do so, first get the diffs by running this command :

```
git --no-pager diff $(git merge-base main HEAD) && git --no-pager diff --no-index /dev/null $(git ls-files --others --exclude-standard)

```

When reviewing the changes, quote the file and the relevant lines in a code block and then explain why you are making this command in a short sentence.
Select only the most important comments that would help improve the code quality, do not comment on everything. Keep maximum 3 comments. No need to mention the positive things, only suggest improvements. Be concise and precise.

The output should look like a list of comments like this :

### Example of one comment :

```diff
# File: .vscode/settings.json
@@ -5,6 +5,7 @@
   "python.analysis.completeFunctionParens": false,
   "python.analysis.inlayHints.functionReturnTypes": true,
   "python.analysis.inlayHints.variableTypes": true,
+  "mypy-type-checker.enabled": false,
   "[python]": {
     "editor.formatOnSave": true,
     "editor.codeActionsOnSave": {
```

Removing the type checking with mypy can reduce the safety of the codebase

### What to check :

In addition to what you would usually check, put a special focus on :

- clarity of the naming
- structure of the file architecture
- length of the file
- readability and easy of comprehension of the modifications
- reasonable testing
