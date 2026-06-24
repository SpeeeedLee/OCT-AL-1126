"""Compact JSON writer (lists stay on one line) -- same style as the source codebase."""
import json


def save_compact_json(data, file_path):
    def format_dict(d, indent=0):
        lines = []
        items = list(d.items())
        for i, (key, value) in enumerate(items):
            comma = '' if i == len(items) - 1 else ','
            if isinstance(value, dict):
                lines.append('  ' * indent + f'"{key}": {{')
                lines.append(format_dict(value, indent + 1))
                lines.append('  ' * indent + '}' + comma)
            elif isinstance(value, list):
                list_str = '[' + ', '.join(json.dumps(x) for x in value) + ']'
                lines.append('  ' * indent + f'"{key}": {list_str}' + comma)
            else:
                lines.append('  ' * indent + f'"{key}": {json.dumps(value)}' + comma)
        return '\n'.join(lines)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('{\n' + format_dict(data, 1) + '\n}')
