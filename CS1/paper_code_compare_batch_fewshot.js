// paper_code_compare_batch_fewshot.js
const fs = require('fs');
const path = require('path');
const { Anthropic } = require('@anthropic-ai/sdk');
const { program } = require('commander');
const { v4: uuidv4 } = require('uuid');

/**
 * Compare a research paper and code file using the Claude API to identify discrepancies.
 * 
 * @param {string} pdfPath - Path to the PDF research paper
 * @param {string} codePath - Path to the code file
 * @param {string} outputDir - Directory to save the output file
 * @param {number} runNumber - The run number for this comparison
 * @param {string} customPrompt - Custom prompt for Claude (uses default if null)
 * @param {string} apiKey - Anthropic API key (defaults to environment variable)
 * @returns {Promise<string>} - Claude's response
 */
async function comparePaperAndCode(pdfPath, codePath, outputDir, runNumber, customPrompt = null, apiKey = null) {
    // Initialize the Anthropic client
    const anthropic = new Anthropic({
        apiKey: process.env.ANTHROPIC_API_KEY || "your-anthropic-key-here"
    });
    
    // Read and encode PDF file as Base64
    const pdfData = fs.readFileSync(pdfPath).toString('base64');
    
    // Read the code file
    const codeContent = fs.readFileSync(codePath, 'utf-8');
    
    // Determine file extension to help Claude understand the language
    const fileExtension = path.extname(codePath).slice(1);
    
    // Extract file names for output naming
    const pdfName = path.basename(pdfPath, path.extname(pdfPath));
    const codeName = path.basename(codePath, path.extname(codePath));
    
    // Default prompt if none provided
    const prompt = customPrompt || (
        `Compare the provided research paper and code implementation to identify any discrepancies that could affect the reproducibility or validity of the work. Focus on differences in methodology, algorithms, mathematical approaches, or key implementation details that might lead to different results or impede replication. Ignore minor deviations in code style, variable names, or superficial details if they are unlikely to change the interpretation or reproducibility of the work.

        After analyzing both artifacts, summarize your findings: For each identified discrepancy, provide a description, references to the relevant paper section(s) and code location(s), and an explanation of how it could affect reproducibility or validity. If no discrepancies are found that have a realistic chance of impacting the reproducibility or validity of the work, simply state that the code implementation accurately represents the paper's described methodology.
        
        Use the following examples as a reference for your decision:
        
        Example 1: Undocumented Layer Name
        Paper Claim: "A ResNet-18 architecture is used for feature extraction."
        Code Section:
        model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], name='backbone')
        Affects Results? No
        Explanation: Layer naming conventions don't alter model functionality or performance.
        
        Example 2: Silent Batch Size Mismatch
        Paper Claim: "Models are trained with a batch size of 64."
        Code Section:
        BATCH_SIZE = 64
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        # Later in training loop:
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            # Paper doesn't mention gradient accumulation
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            if (i+1) % 2 == 0: # Effective batch size = 128
                optimiser.step()
                optimiser.zero_grad()
        Affects Results? Yes
        Explanation: Gradient accumulation alters effective batch size (64 → 128), affecting optimisation dynamics.
        
        Example 3: Redundant Identity Layer
        Paper Claim: "The transformer uses 8 attention heads."
        Code Section:
        # Extra identity layer that does nothing
        class TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MultiheadAttention(embed_dim=512, num_heads=8)
                self.norm = nn.Identity() # Paper doesn't mention this
        Affects Results? No
        Explanation: Identity layers don't alter model behavior despite being undocumented.
        
        Example 4: Wrong Value
        Paper Claim: "Gradients are clipped at 1.0 during training."
        Code Section:
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.1) # Paper claims 1.0
        Affects Results? Yes
        Explanation: Aggressive clipping (0.1 vs. 1.0) stifles gradient updates, altering training dynamics.
        
        Example 5: System Architecture
        Paper claim: "We used a convolutional neural network with 3 convolutional layers and 2 fully connected layers."
        Code section:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
        Affects results?: No
        Explanation: The overall architecture matches the paper's description. These minor deviations are unlikely to significantly affect the results, as the fundamental methodology remains the same, unless the paper mentions specific values for each later.
        
        If none of the discrepancies are found to impact the results, respond simply with "NO MAJOR DISCREPANCIES FOUND" followed by a short message saying that the code is a faithful implementation of the paper.`   );
    
    // Generate a unique conversation ID to prevent history issues
    const conversationId = uuidv4();
    
    console.log(`Analyzing paper "${pdfName}" with code "${codeName}"`);
    
    try {
        // Send request to Claude API
        const response = await anthropic.messages.create({
            model: "claude-3-7-sonnet-20250219",
            max_tokens: 1024,
            temperature: 0.5,
            system: "This is a new conversation with ID: " + conversationId,
            messages: [
                {
                    role: "user",
                    content: [
                        {
                            type: "document",
                            source: {
                                type: "base64",
                                media_type: "application/pdf",
                                data: pdfData
                            }
                        },
                        {
                            type: "text", 
                            text: `I'm also providing a ${fileExtension} code file:\n\n\`\`\`${fileExtension}\n${codeContent}\n\`\`\`\n\n${prompt}`
                        }
                    ]
                }
            ]
        });
        
        // Get Claude's response text
        const result = response.content[0].text;
        
        // Save result to markdown file
        const outputFileName = `${codeName}_Run${runNumber}.md`;
        const outputPath = path.join(outputDir, outputFileName);
        
        // Create markdown content
        const mdContent = `# Paper-Code Consistency Analysis

**Paper:** ${pdfName}
**Code:** ${codeName}
**Analysis Date:** ${new Date().toISOString().split('T')[0]}

## Analysis Results

${result}`;
        
        fs.writeFileSync(outputPath, mdContent);
        
        console.log(`Analysis saved to: ${outputPath}`);
        return result;
    } catch (error) {
        console.error('Error comparing paper and code:', error);
        if (error.response) {
            console.error('API Error Details:', error.response.data);
        }
        throw error;
    }
}

/**
 * Run a batch of comparisons between a paper and code file multiple times.
 * 
 * @param {string} pdfPath - Path to the PDF research paper
 * @param {string} codePath - Path to the code file
 * @param {string} outputDir - Directory to save the output file
 * @param {number} numRuns - Number of times to run each comparison (default: 3)
 * @param {string} customPrompt - Custom prompt for Claude (uses default if null)
 * @param {string} apiKey - Anthropic API key (defaults to environment variable)
 * @returns {Promise<void>}
 */
async function runComparisonsInBatch(pdfPath, codePath, outputDir, numRuns = 3, customPrompt = null, apiKey = null) {
    for (let i = 1; i <= numRuns; i++) {
        console.log(`Running comparison ${i} of ${numRuns}...`);
        await comparePaperAndCode(pdfPath, codePath, outputDir, i, customPrompt, apiKey);
    }
}

// Command line interface setup
if (require.main === module) {
    program
        .name('paper-code-compare-batch')
        .description('Compare a research paper and code with Claude AI in batch mode')
        .version('1.0.0')
        .requiredOption('-p, --pdf <path>', 'Path to the PDF research paper')
        .requiredOption('-c, --code <path>', 'Path to the code file to analyze')
        .option('-o, --output-dir <path>', 'Directory to save output files', './Results Few Shot')
        .option('-r, --runs <number>', 'Number of times to run each comparison', 3)
        .option('--prompt <text>', 'Custom prompt for Claude (uses default if not provided)')
        .option('--api-key <key>', 'Anthropic API key (optional, can use environment variable)')
        .action(async (options) => {
            try {
                // Ensure output directory exists
                if (!fs.existsSync(options.outputDir)) {
                    fs.mkdirSync(options.outputDir, { recursive: true });
                }
                
                // Run the comparisons in batch
                await runComparisonsInBatch(
                    options.pdf,
                    options.code,
                    options.outputDir,
                    options.runs,
                    options.prompt,
                    options.apiKey
                );
                
                console.log(`\nAll comparisons completed. Results saved in ${options.outputDir}`);
            } catch (error) {
                console.error(`Error: ${error.message}`);
                process.exit(1);
            }
        });

    program.parse();
}

module.exports = { comparePaperAndCode, runComparisonsInBatch };