# TheEmperorsNewCode_Thesis

This repository contains the relevant files used for comparing research papers with their corresponding code implementations to asses reproducibility and identify discrepancies. The CS1 folder contains the relevant materials for Case Study 1, and the CS2 folder contains the relevant materials for Case Study 2.

## Overview

The tools use various LLMs (Claude, OpenAI GPT, Gemini) to analyse research papers (PDF format) against accompanying code implementations, identifying potential discrepancies that could affect reproducibility. 

## Prerequisites
**Required Dependencies**
Install the following Node.js packages:

```bash
npm install @anthropic-ai/sdk commander uuid
npm install openai  # For OpenAI experiments
npm install @google/generative-ai  # For Gemini experiments
npm install adm-zip  # For CS2 experiments (zip handling)
npm install dotenv  # Optional, for environment variables
```

**API Keys**
Set up your API keys as environment variables in a .env file. Set the location such that the javascript file used to execute the comparison can access it within the folder:

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export OPENAI_API_KEY="your-openai-api-key" 
export GOOGLE_API_KEY="your-google-api-key"
```
Alternatively, you can hardcode them in the respective files or pass them via command line arguments.

## CS1 Experiments - A Baseline Test with single files
CS1 experiments compare a research paper (PDF) with a single code file. The original MNIST.py code to run this experiment is based upon the MNIST digit classification from [machinelearningmastery.com](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/). If you wish to actually run the MNIST.py code, follow his instructions for setup (downloading Keras and Tensorflow) and execution.

### Available Scripts
**paper_code_compare_batch.js** - Basic comparison with default prompting

**paper_code_compare_batch_fewshot.js** - Enhanced with few-shot examples for better classification

### How to use
**Zero-Shot Comparison**
For this experiment we use the spoof paper 'DeepCNN_JournalArticle.pdf' and the discrepancy files within "Discrepancy Files" or "False Discrepancies". The default runs is set to 3, and the output folder is 'Results', though this is configurable.

```bash
# Basic usage
node paper_code_compare_batch.js -p DeepCNN_JournalArticle.pdf -c "Discrepancy Files/MNIST_DP_01.py"

# Specify output directory and number of runs
node paper_code_compare_batch.js -p paper.pdf -c implementation.py -o ./your own folder name -r 3

# Provide API key directly
node paper_code_compare_batch.js -p paper.pdf -c code.py --api-key your-api-key-here
```

**Few-Shot Comparison**
The default runs is set to 3, and the output folder is 'Results Few-Shot', though this is configurable.

```bash
# Same usage as basic, but with enhanced prompting
node paper_code_compare_batch_fewshot.js -p DeepCNN_JournalArticle.pdf -c "Discrepancy Files/MNIST_DP_01.py" -o ./results_fewshot
```

**CS1 Command Line Options**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--pdf` | `-p` | Path to PDF research paper | **Required** |
| `--code` | `-c` | Path to code file | **Required**, within 'Discrepancy Files' or 'False Discrepancies' |
| `--output-dir` | `-o` | Output directory for results | `./Results` or `./Results Few Shot` or configure to your own |
| `--runs` | `-r` | Number of analysis runs | `3` |
| `--prompt` | | Custom prompt (optional) | Uses default |
| `--api-key` | | Anthropic API key | Uses environment variable |

## CS2 Experiments - Implementing New Framework

CS2 experiments compare research papers with code archives (zip files) downloaded from GitHub. Many of the experiments are with Claude, changing the prompt strategy and and settings. Other javascript files are for the comparison between models. The SliceGAN paper and Zip archives with deliberately inserted errors are located within this folder.

### Available Scripts

1. **CS2_test_new_prompt.js** - Basic Claude analysis with 'New Perspective' framework prompt
2. **CS2_extended_thinking_test.js** - Claude with extended thinking capabilities, configurable to change the model between Sonnet and Opus, and to determine token allocation.
3. **CS2_PaperThenCode_input.js** - Two-stage analysis (paper first, then code)
4. **CS2_CodeThenPaper_input.js** - Two-stage analysis (code first, then paper)
5. **CS2_GPT_model_test.js** - OpenAI analysis, configure to o4-mini/o3/future models
6. **CS2_gemini_model_test.js** - Google Gemini analysis, configure to Pro/Flash/future models

### How to use

#### Basic Claude Analysis
Output results folder set to "CS2 Results Output\CS2 new prompt results" by default. Configure these settings how you wish.

```bash
node CS2_test_new_prompt.js -p SliceGAN-paper.pdf -z SliceGAN-master-anisotropic //Example usage of SliceGAN comparison
```

#### Extended Thinking Analysis (Claude Opus 4)
Output results folder set to "CS2 Results Output/CS2 Opus 4" by default. The output from extended thinking experiments with Sonnet 3.7 are under "CS2 Results Output/CS2 extended thinking results"

```bash
node CS2_extended_thinking_test.js -p paper.pdf -z codebase.zip
```
You may wish to change the model or the token allocation, as I did for Case Study 2. These settings can be changed in lines 94-100. **Be sure to change the output folder before you run new experiments with the new API settings**

```javascript
// Send request to Claude API
        const response = await anthropic.messages.create({
            model: "claude-opus-4-20250514", //Configure to desired model
            max_tokens: 20000, //Configure max tokens as needed
            thinking: {
                type: "enabled",
                budget_tokens: 16000 //Change thinking tokens here tokens as needed
            }
```

#### Two-Stage Analysis (Paper → Code)
Output results folder set to "CS2 Results Output/CS2 PaperThenCode results" by default. 

```bash
node CS2_PaperThenCode_input.js -p paper.pdf -z codebase.zip
```

#### Two-Stage Analysis (Code → Paper)
Output results folder set to "CS2 Results Output/CS2 CodeThenPaper results" by default.

```bash
node CS2_CodeThenPaper_input.js -p paper.pdf -z codebase.zip
```

#### OpenAI GPT Analysis
Model set to o4-mini by default, as is results folder set to "CS2 Results Output/CS2 o4 mini results".
```bash
node CS2_GPT_model_test.js -p paper.pdf -z codebase.zip
```
The model is specified at the top of this folder in line 27. Change the token settings in lines 81-86.

```javascript
const modelName = "o4-mini"; // change to 'o3' or any other model
...
...
const response = await client.responses.create({
            model: modelName, //configurable at top of the page
            max_output_tokens:20000, 
            reasoning: { 
                "effort": "high" //most equivalent option to 16000
            },
```

#### Gemini Analysis
Model set to Gemini 2.5 Pro by default, as is results folder set to "CS2 Results Output/CS2 Gemini 2.5 Pro results".
```bash
node CS2_gemini_model_test.js -p paper.pdf -z codebase.zip
```
The model is specified in line 29. Change this how you wish. Gemini does not yet allow you to override thinking token allocation yourself.
```javascript
const model = genAI.getGenerativeModel({ model: "gemini-2.5-pro-preview-05-06" }); // Configure to your desired model
```

### CS2 Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--pdf` | `-p` | Path to PDF research paper | **Required** |
| `--zip` | `-z` | Path to zip archive with code | **Required**, using SliceGAN-master-anisotropic, SliceGAN-master-training, etc. for this |
| `--output-dir` | `-o` | Output directory for results | Varies by script |
| `--runs` | `-r` | Number of analysis runs | `3` |
| `--prompt` | | Custom prompt (optional) | Uses default |
| `--api-key` | | API key for respective service | Uses environment variable |

## File Structure Requirements

### CS1 Input Files
- **PDF**: Research paper in PDF format
- **Code**: Single code file (.py, .js, .cpp, .java, .r, etc.)

### CS2 Input Files
- **PDF**: Research paper in PDF format  
- **ZIP**: Archive containing:
  - Code files (.py, .js, .cpp, .java, .r, etc.)
  - README files (README.md, readme.txt, etc.)
  - Project structure

## Output Format

All experiments generate markdown files with the following structure:

```markdown
# Paper-Code Consistency Analysis

**Paper:** [paper_name]
**Code:** [code_name] 
**Analysis Date:** [YYYY-MM-DD]

## Analysis Results

[AI model's analysis and findings]
```

### Output Naming Convention

- **CS1**: `{codename}_Run{number}.md`
- **CS2**: `{zipname}_Run{number}_{model}.md`

## Model Configuration

### Claude Models
- **Default**: `claude-3-7-sonnet-20250219`
- **Extended Thinking**: `claude-opus-4-20250514` - used in final model comparison tests

### OpenAI Models  
- **Default**: `o4-mini`

### Gemini Models
- **Default**: `gemini-2.5-pro-preview-05-06`

## Semantic Chunking (CS2)

CS2 experiments use `semanticChunking.js` to intelligently parse code archives:

- **Header/Imports**: Grouped separately
- **Classes**: Each class as a semantic unit
- **Functions**: Each function as a semantic unit  
- **Other Code**: Remaining code grouped logically
- **README**: Processed separately from code

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure environment variables are set or provide keys via `--api-key`
2. **File Not Found**: Verify paths to PDF and code files are correct
3. **Permission Errors**: Ensure write permissions for output directories
4. **Large Files**: Some models have token limits; consider splitting large codebases

### Debug Mode

Add error logging by checking the console output. Each script provides detailed error messages and API response information.

## Batch Processing

To run multiple experiments:

```bash
# Run different models on the same inputs
node CS2_test_new_prompt.js -p paper.pdf -z code.zip -o ./claude_results
node CS2_GPT_model_test.js -p paper.pdf -z code.zip -o ./gpt_results  
node CS2_gemini_model_test.js -p paper.pdf -z code.zip -o ./gemini_results

# Run multiple papers with the same code
for paper in papers/*.pdf; do
    node CS2_test_new_prompt.js -p "$paper" -z codebase.zip -o ./batch_results
done
```

## Disclaimer

These tools are provided for research purposes. AI model outputs should be critically evaluated and not taken as definitive assessments of code quality or paper accuracy.





