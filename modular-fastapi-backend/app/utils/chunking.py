from typing import List, Optional
import re

def chunk_text_fixed(text: str, chunk_size: int = 500) -> List[str]:
    """
    Naive fixed-size chunking by whitespace tokens (approximate tokens).
    Splits text into chunks of chunk_size tokens.
    """
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = " ".join(tokens[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def chunk_text_sentences(text: str, chunk_size: int = 500) -> List[str]:
    """
    Sentence-split chunking: group sentences until approximate chunk_size tokens reached.
    """
    # very simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_count = 0
    for s in sentences:
        toks = s.split()
        if current_count + len(toks) > chunk_size and current:
            chunks.append(" ".join(current))
            current = [s]
            current_count = len(toks)
        else:
            current.append(s)
            current_count += len(toks)
    if current:
        chunks.append(" ".join(current))
    return chunks

def chunk_text_recursive(
    text: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Recursive character text splitter with overlap support.
    Tries to split on multiple separators in order of preference:
    1. Paragraph breaks (double newlines)
    2. Sentence endings (. ! ?)
    3. Sentence separators (; :)
    4. Word boundaries (spaces)
    5. Characters (fallback)
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens (approximate, using whitespace)
        chunk_overlap: Number of tokens to overlap between chunks
        separators: Custom list of separators (optional)
    
    Returns:
        List of text chunks with overlap
    """
    if separators is None:
        separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Single newlines
            ". ",    # Sentence endings
            "! ", 
            "? ",
            "; ",    # Sentence separators
            ": ",
            " ",     # Word boundaries
            "",      # Character-level (fallback)
        ]
    
    # Normalize text
    text = text.strip()
    if not text:
        return []
    
    # Split by the first separator
    separator = separators[0]
    splits = _split_text(text, separator)
    
    # If we have more separators and splits are too large, recurse
    if len(separators) > 1:
        chunks = []
        for split in splits:
            if _count_tokens(split) > chunk_size:
                # Recurse with remaining separators
                sub_chunks = chunk_text_recursive(
                    split, 
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap,
                    separators=separators[1:]
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(split)
    else:
        chunks = splits
    
    # Merge chunks with overlap
    return _merge_chunks_with_overlap(chunks, chunk_size, chunk_overlap)

def _split_text(text: str, separator: str) -> List[str]:
    """Split text by separator, preserving the separator in the splits."""
    if separator == "":
        # Character-level split
        return list(text)
    splits = text.split(separator)
    # Add separator back to all but the last split
    result = []
    for i, split in enumerate(splits):
        if i < len(splits) - 1:
            result.append(split + separator)
        else:
            result.append(split)
    return [s for s in result if s.strip()]  # Remove empty splits

def _count_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting."""
    return len(text.split())

def _merge_chunks_with_overlap(
    chunks: List[str], 
    chunk_size: int, 
    chunk_overlap: int
) -> List[str]:
    """Merge chunks ensuring they don't exceed chunk_size, with overlap between chunks."""
    if not chunks:
        return []
    
    merged = []
    current_chunk = []
    current_size = 0
    
    for chunk in chunks:
        chunk_tokens = _count_tokens(chunk)
        
        # If adding this chunk would exceed size, finalize current chunk
        if current_size + chunk_tokens > chunk_size and current_chunk:
            merged_text = " ".join(current_chunk)
            merged.append(merged_text)
            
            # Start new chunk with overlap from previous
            if chunk_overlap > 0:
                # Get last N tokens from previous chunk for overlap
                prev_tokens = merged_text.split()
                overlap_tokens = prev_tokens[-chunk_overlap:] if len(prev_tokens) > chunk_overlap else prev_tokens
                current_chunk = overlap_tokens + [chunk]
                current_size = len(overlap_tokens) + chunk_tokens
            else:
                current_chunk = [chunk]
                current_size = chunk_tokens
        else:
            current_chunk.append(chunk)
            current_size += chunk_tokens
    
    # Add final chunk
    if current_chunk:
        merged.append(" ".join(current_chunk))
    
    return merged