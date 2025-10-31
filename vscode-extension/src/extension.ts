/**
 * StatLang VS Code Extension
 * 
 * This extension provides StatLang syntax highlighting and execution support
 * for .statlang files using a Python backend.
 */

import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as os from 'os';

let outputChannel: vscode.OutputChannel;
let logChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
    console.log('StatLang extension is now active!');
    
    // Create output channels
    outputChannel = vscode.window.createOutputChannel('StatLang Output');
    logChannel = vscode.window.createOutputChannel('StatLang Log');
    
    // Register commands
    const runFileCommand = vscode.commands.registerCommand('statlang.runFile', runFile);
    const runSelectionCommand = vscode.commands.registerCommand('statlang.runSelection', runSelection);
    const checkSyntaxCommand = vscode.commands.registerCommand('statlang.checkSyntax', checkSyntax);
    
    context.subscriptions.push(runFileCommand, runSelectionCommand, checkSyntaxCommand);
    
    // Note: VS Code notebooks use the Jupyter kernel directly
    // The StatLang kernel is already installed and should be available
    // in VS Code's kernel selector for .ipynb files
    
    // Show output channel on first activation
    outputChannel.show();
}

export function deactivate() {
    // Clean up resources
    if (outputChannel) {
        outputChannel.dispose();
    }
    if (logChannel) {
        logChannel.dispose();
    }
}

/**
 * Run the current .statlang file
 */
async function runFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor found.');
        return;
    }
    
    const document = editor.document;
    if (!document.fileName.endsWith('.statlang')) {
        vscode.window.showErrorMessage('Current file is not a .statlang file.');
        return;
    }
    
    // Save the file first
    await document.save();
    
    // Execute the file
    await executeStatLangFile(document.fileName);
}

/**
 * Run the selected text in the current .statlang file
 */
async function runSelection() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor found.');
        return;
    }
    
    const selection = editor.selection;
    if (selection.isEmpty) {
        vscode.window.showErrorMessage('No text selected.');
        return;
    }
    
    const selectedText = editor.document.getText(selection);
    await executeStatLangText(selectedText);
}

/**
 * Check syntax of the current .statlang file
 */
async function checkSyntax() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor found.');
        return;
    }
    
    const document = editor.document;
    if (!document.fileName.endsWith('.statlang')) {
        vscode.window.showErrorMessage('Current file is not a .statlang file.');
        return;
    }
    
    // For now, just run the file in syntax-check mode
    // In a full implementation, this would parse without executing
    vscode.window.showInformationMessage('Syntax check not yet implemented. Running file instead.');
    await runFile();
}

/**
 * Execute a .statlang file
 */
async function executeStatLangFile(filePath: string) {
    outputChannel.clear();
    outputChannel.appendLine(`Executing: ${filePath}`);
    outputChannel.appendLine('='.repeat(50));
    
    try {
        const pythonPath = await getPythonPath();
        const runnerPath = await getStatLangRunnerPath();
        
        // Spawn Python process to run StatLang
        const args = ['-u', runnerPath, filePath];
        const process = spawn(pythonPath, args, {
            cwd: path.dirname(filePath),
            stdio: ['pipe', 'pipe', 'pipe']
        });
        
        await handleProcessOutput(process, filePath);
        
    } catch (error) {
        const errorMessage = `Error executing ${filePath}: ${error}`;
        outputChannel.appendLine(errorMessage);
        vscode.window.showErrorMessage(errorMessage);
    }
}

/**
 * Execute StatLang text directly
 */
async function executeStatLangText(statlangText: string) {
    outputChannel.clear();
    outputChannel.appendLine('Executing selected StatLang code:');
    outputChannel.appendLine('='.repeat(50));
    
    try {
        const pythonPath = await getPythonPath();
        const runnerPath = await getStatLangRunnerPath();
        
        // Create a temporary file with the selected text
        const tempFile = path.join(os.tmpdir(), `statlang_temp_${Date.now()}.statlang`);
        require('fs').writeFileSync(tempFile, statlangText);
        
        // Spawn Python process
        const args = ['-u', runnerPath, tempFile];
        const process = spawn(pythonPath, args, {
            stdio: ['pipe', 'pipe', 'pipe']
        });
        
        await handleProcessOutput(process, 'Selected Code');
        
        // Clean up temporary file
        require('fs').unlinkSync(tempFile);
        
    } catch (error) {
        const errorMessage = `Error executing StatLang code: ${error}`;
        outputChannel.appendLine(errorMessage);
        vscode.window.showErrorMessage(errorMessage);
    }
}

/**
 * Handle process output and errors
 */
async function handleProcessOutput(process: ChildProcess, source: string): Promise<void> {
    return new Promise((resolve, reject) => {
        let hasError = false;
        
        // Handle stdout
        process.stdout?.on('data', (data) => {
            const output = data.toString();
            outputChannel.append(output);
        });
        
        // Handle stderr
        process.stderr?.on('data', (data) => {
            const error = data.toString();
            logChannel.append(`[${source}] ${error}`);
            hasError = true;
        });
        
        // Handle process completion
        process.on('close', (code) => {
            if (code === 0) {
                outputChannel.appendLine(`\nExecution completed successfully.`);
                resolve();
            } else {
                const errorMessage = `Process exited with code ${code}`;
                outputChannel.appendLine(`\n${errorMessage}`);
                if (hasError) {
                    logChannel.show();
                }
                reject(new Error(errorMessage));
            }
        });
        
        // Handle process errors
        process.on('error', (error) => {
            const errorMessage = `Process error: ${error.message}`;
            outputChannel.appendLine(errorMessage);
            reject(error);
        });
        
        // Show output channel
        outputChannel.show();
    });
}

/**
 * Get Python executable path
 */
async function getPythonPath(): Promise<string> {
    const config = vscode.workspace.getConfiguration('statlang');
    let pythonPath = config.get<string>('pythonPath');
    
    if (!pythonPath) {
        // Try to find Python in common locations
        const pythonCommands = ['python3', 'python'];
        
        for (const cmd of pythonCommands) {
            try {
                await executeCommand(cmd, ['--version']);
                pythonPath = cmd;
                break;
            } catch {
                // Continue to next command
            }
        }
        
        if (!pythonPath) {
            throw new Error('Python not found. Please install Python or set the statlang.pythonPath setting.');
        }
    }
    
    return pythonPath;
}

/**
 * Get StatLang runner script path
 */
async function getStatLangRunnerPath(): Promise<string> {
    const config = vscode.workspace.getConfiguration('statlang');
    let runnerPath = config.get<string>('runtimePath');
    
    if (!runnerPath) {
        // Always use the installed Python package directly
        // This avoids version-specific path issues
        try {
            await executeCommand('python', ['-c', 'import stat_lang; print("OK")']);
            runnerPath = 'python -m stat_lang';
        } catch {
            throw new Error('StatLang runtime not found. Please install the package or set the statlang.runtimePath setting.');
        }
    }
    
    return runnerPath;
}

/**
 * Execute a command and return a promise
 */
function executeCommand(command: string, args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
        const process = spawn(command, args);
        let output = '';
        let error = '';
        
        process.stdout?.on('data', (data) => {
            output += data.toString();
        });
        
        process.stderr?.on('data', (data) => {
            error += data.toString();
        });
        
        process.on('close', (code) => {
            if (code === 0) {
                resolve(output);
            } else {
                reject(new Error(error || `Command failed with code ${code}`));
            }
        });
        
        process.on('error', (err) => {
            reject(err);
        });
    });
}
