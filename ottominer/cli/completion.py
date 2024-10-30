from .analyzers import get_available_analyzers
from pathlib import Path

def generate_completion(shell: str = 'bash') -> str:
    """Generate shell completion script"""
    analyzers = get_available_analyzers()
    
    if shell == 'bash':
        return _generate_bash_completion(analyzers)
    elif shell == 'zsh':
        return _generate_zsh_completion(analyzers)
    else:
        raise ValueError(f"Unsupported shell: {shell}")

def install_completion(shell: str = 'bash') -> None:
    """Install shell completion script"""
    script = generate_completion(shell)
    
    if shell == 'bash':
        completion_path = Path.home() / '.bash_completion.d' / 'ottominer'
    elif shell == 'zsh':
        completion_path = Path.home() / '.zsh' / 'completion' / '_ottominer'
    else:
        raise ValueError(f"Unsupported shell: {shell}")
        
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.write_text(script)

def _generate_bash_completion(analyzers: list) -> str:
    """Generate Bash completion script"""
    return f"""
_ottominer_completion() {{
    local cur prev opts
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    opts="--help -h -i --input -o --output --extraction-mode --batch-size --workers"
    
    if [[ $prev == "--extraction-mode" ]]; then
        COMPREPLY=( $(compgen -W "simple ocr hybrid" -- $cur) )
        return 0
    fi
    
    if [[ $prev == "--type" ]]; then
        COMPREPLY=( $(compgen -W "{' '.join(analyzers)}" -- $cur) )
        return 0
    fi
    
    COMPREPLY=( $(compgen -W "$opts" -- $cur) )
}}

complete -F _ottominer_completion ottominer
"""

def _generate_zsh_completion(analyzers: list) -> str:
    """Generate Zsh completion script"""
    return f"""
#compdef ottominer

_arguments \\
  '--help[Show help message]' \\
  '-h[Show help message]' \\
  '-i[Input directory]:filename:_files -/' \\
  '--input[Input directory]:filename:_files -/' \\
  '-o[Output directory]:filename:_files -/' \\
  '--output[Output directory]:filename:_files -/' \\
  '--extraction-mode[Text extraction mode]:(simple ocr hybrid)' \\
  '--batch-size[Batch size]:number' \\
  '--workers[Number of workers]:number' \\
  '--type[Analysis type]:({' '.join(analyzers)})'
"""