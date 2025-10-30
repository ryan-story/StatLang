/**
 * StatLang Notebook Provider for VS Code
 *
 * This module provides notebook support for StatLang,
 * allowing interactive execution of StatLang code in VS Code notebooks.
 */

import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';

export class OSASNotebookProvider {
    private interpreters: Map<string, any> = new Map();
    private outputChannel: vscode.OutputChannel;

    constructor() {
        this.outputChannel = vscode.window.createOutputChannel('StatLang Notebook');
    }

    async resolveNotebook(document: vscode.NotebookDocument, webview: any): Promise<void> {
        // Initialize notebook communication
        this.outputChannel.appendLine(`Resolved notebook: ${document.uri.fsPath}`);
    }

    async executeCell(document: vscode.NotebookDocument, cell: vscode.NotebookCell): Promise<void> {
        const statlangCode = cell.document.getText();
        
        if (!statlangCode.trim()) {
            // Empty cell - clear outputs
            return;
        }

        try {
            // Execute StatLang code
            const result = await this.executeStatLangCode(statlangCode, document.uri.fsPath);
            
            // Create output items
            const outputItems: vscode.NotebookCellOutputItem[] = [];
            
            // Add text output
            if (result.output) {
                outputItems.push(vscode.NotebookCellOutputItem.text(result.output, 'text/plain'));
            }
            
            // Add error output
            if (result.errors) {
                outputItems.push(vscode.NotebookCellOutputItem.text(result.errors, 'text/plain'));
            }
            
            // Add dataset displays
            if (result.datasets) {
                for (const [name, dataset] of Object.entries(result.datasets)) {
                    const html = this.createDatasetHTML(name, dataset as any);
                    outputItems.push(vscode.NotebookCellOutputItem.text(html, 'text/html'));
                }
            }
            
            // Add PROC results
            if (result.proc_results) {
                for (const procResult of result.proc_results) {
                    const html = this.createPROCResultHTML(procResult);
                    outputItems.push(vscode.NotebookCellOutputItem.text(html, 'text/html'));
                }
            }
            
            // Update cell output - Note: outputs are read-only, this would need to be handled differently
            // For now, we'll just log the results
            if (outputItems.length > 0) {
                this.outputChannel.appendLine(`Cell execution completed with ${outputItems.length} output items`);
            }
            
        } catch (error) {
            // Handle execution error
            this.outputChannel.appendLine(`Error executing StatLang code: ${error}`);
        }
    }

    private async executeStatLangCode(code: string, notebookPath: string): Promise<any> {
        return new Promise((resolve, reject) => {
            const pythonPath = this.getPythonPath();
            const scriptPath = path.join(__dirname, '..', '..', 'python', 'notebook_runner.py');
            
            const process = spawn(pythonPath, [scriptPath, '--code', code, '--notebook', notebookPath], {
                cwd: path.dirname(notebookPath),
                stdio: ['pipe', 'pipe', 'pipe']
            });

            let output = '';
            let error = '';

            process.stdout.on('data', (data) => {
                output += data.toString();
            });

            process.stderr.on('data', (data) => {
                error += data.toString();
            });

            process.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(output);
                        resolve(result);
                    } catch (e) {
                        resolve({
                            output: output,
                            errors: error,
                            datasets: {},
                            proc_results: []
                        });
                    }
                } else {
                    reject(new Error(`Process exited with code ${code}: ${error}`));
                }
            });

            process.on('error', (err) => {
                reject(err);
            });
        });
    }

    private getPythonPath(): string {
        const config = vscode.workspace.getConfiguration('statlang');
        return (config.get('pythonPath') as string) || 'python';
    }

    private createDatasetHTML(name: string, dataset: any): string {
        const { shape, columns, head } = dataset;
        
        let html = `
            <div class="statlang-dataset" style="margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; padding: 10px; font-family: monospace;">
                <h4 style="margin: 0 0 10px 0; color: #333;">Dataset: ${name}</h4>
                <p style="margin: 0 0 10px 0; color: #666; font-size: 0.9em;">
                    ${shape[0]} observations, ${shape[1]} variables
                </p>
        `;

        if (head && head.length > 0) {
            html += `
                <table style="border-collapse: collapse; width: 100%; font-size: 0.9em;">
                    <thead>
                        <tr style="background-color: #f5f5f5;">
                            ${columns.map((col: string) => `<th style="border: 1px solid #ddd; padding: 8px; text-align: left;">${col}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${head.map((row: any) => `<tr>${columns.map((col: string) => `<td style="border: 1px solid #ddd; padding: 8px;">${row[col] || ''}</td>`).join('')}</tr>`).join('')}
                    </tbody>
                </table>
            `;
        }

        html += '</div>';
        return html;
    }

    private createPROCResultHTML(procResult: any): string {
        // Create HTML for results
        let html = `
            <div class="statlang-proc-result" style="margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; padding: 10px;">
                <h4 style="margin: 0 0 10px 0; color: #333;">${procResult.proc_name} Results</h4>
        `;

        if (procResult.data) {
            html += `
                <table style="border-collapse: collapse; width: 100%; font-size: 0.9em;">
                    <tbody>
                        ${procResult.data.map((row: any) => `<tr>${Object.values(row).map((val: any) => `<td style=\"border: 1px solid #ddd; padding: 8px;\">${val}</td>`).join('')}</tr>`).join('')}
                    </tbody>
                </table>
            `;
        }

        html += '</div>';
        return html;
    }
}
