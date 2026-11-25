"""Query-scoped Entity, Hashtag, Proper-Noun, Mention, URL & Year Anonymizer.

Used in Stage 2B to stress-test whether SUG-Meso's gains rely on LLM pretraining
memorization of historical named entities, hashtags, or dates (Event2012 / Event2018),
while holding structural clustering, query routing, and causal token budgets fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
import threading
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sug_sed_method.core.data_loader import SocialDataset
from sug_sed_method.method.llm_client import build_pairwise_same_event_prompt, build_post_assign_prompt
from sug_sed_method.method.star_meso import ClusterSummary


@dataclass
class QueryItem:
    query_id: str
    query_type: str
    primary_nodes: List[int]
    candidate_clusters: List[int]
    merge_c1: Optional[int] = None
    merge_c2: Optional[int] = None
    merge_c1_medoids: List[int] = None  # type: ignore[assignment]
    merge_c2_medoids: List[int] = None  # type: ignore[assignment]
    seam_cluster_id: Optional[int] = None
    seam_sub1_medoids: List[int] = None  # type: ignore[assignment]
    seam_sub2_medoids: List[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.merge_c1_medoids is None:
            self.merge_c1_medoids = []
        if self.merge_c2_medoids is None:
            self.merge_c2_medoids = []
        if self.seam_sub1_medoids is None:
            self.seam_sub1_medoids = []
        if self.seam_sub2_medoids is None:
            self.seam_sub2_medoids = []


# Boilerplate uppercase tokens & EN/FR sentence-internal capitalized words to avoid
# masking as proper nouns so relational/event syntax stays natural.
_UPPER_BOILERPLATE: Set[str] = {
    "RT", "MT", "VIA", "LIVE", "INFO", "DIRECT", "URGENT", "VIDEO", "PHOTO",
    "BREAKING", "UPDATE", "RIP", "NO", "YES", "OK", "EN", "FR", "VS", "AM", "PM",
    "TODAY", "NOW", "NEW", "WATCH", "READ", "JUST", "MORE", "ALL", "THE",
}

_CAP_FUNCTION_WORDS: Set[str] = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "but", "or", "if",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might", "can", "not",
    "this", "that", "these", "those", "it", "its", "he", "she", "they", "we", "you",
    "his", "her", "their", "our", "your", "who", "what", "when", "where", "why", "how",
    "after", "before", "over", "under", "into", "from", "with", "about", "against",
    "between", "during", "while", "more", "most", "new", "live", "breaking", "update",
    "video", "photo", "watch", "read", "just", "now", "today", "says", "said",
    # French function words
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l", "au", "aux", "en",
    "dans", "sur", "sous", "pour", "par", "avec", "sans", "et", "ou", "mais", "donc",
    "qui", "que", "quoi", "dont", "il", "elle", "ils", "elles", "on", "nous", "vous",
    "ce", "cet", "cette", "ces", "son", "sa", "ses", "leur", "leurs", "apres", "avant",
    "contre", "entre", "pendant", "plus", "moins", "tres", "tout", "tous", "toute",
    "est", "sont", "etait", "ete", "avoir", "fait", "faire", "dit", "selon", "direct",
}

_HEADLINE_COMMON_WORDS: Set[str] = {
    "stripped", "titles", "ratified", "report", "missing", "child", "children",
    "female", "male", "white", "black", "global", "warning", "suspect", "suspects",
    "found", "dead", "police", "protesters", "breaking", "clash", "saints", "tribute",
    "death", "interview", "exclusif", "direct", "live", "urgent", "update", "news",
    "photos", "video", "link", "watch", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "lundi", "mardi", "mercredi", "jeudi", "vendredi",
    "samedi", "dimanche", "journalist", "producer", "election", "count", "rally",
    "opposition", "government", "official", "scandal", "files", "over", "amid",
    "during", "after", "against", "before", "under", "more", "says", "said",
}

_AUDIT_STOPWORDS: Set[str] = _CAP_FUNCTION_WORDS | _HEADLINE_COMMON_WORDS | {
    "entity", "hashtag", "user", "url", "year", "group", "subgroup", "cluster",
    "post", "event", "incident", "people", "match", "game", "team", "minister",
    "mort", "morts", "tue", "equipe", "gouvernement", "ministre", "infos",
}

_ACCENT_EQUIV: Dict[str, str] = {
    "a": "[aàáâãäå]",
    "c": "[cç]",
    "e": "[eéèêë]",
    "i": "[iìíîï]",
    "n": "[nñ]",
    "o": "[oòóôõö]",
    "u": "[uùúûü]",
    "y": "[yýÿ]",
}

_WORD_CH = r"A-Za-z0-9_\u00C0-\u024F"
_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_MENTION_RE = re.compile(rf"@[{_WORD_CH}]+")
_HASHTAG_RE = re.compile(rf"#[\w\u00C0-\u024F]+")
_YEAR_RE = re.compile(r"\b(?:19\d{2}|20[0-2]\d)\b")
_MULTI_CAP_RE = re.compile(
    rf"(?<![{_WORD_CH}])"
    r"([A-Z\u00C0-\u00DE][A-Za-z0-9\u00C0-\u024F'’\-]{1,}(?:\s+[A-Z\u00C0-\u00DE][A-Za-z0-9\u00C0-\u024F'’\-]{1,}){1,2})"
    rf"(?![{_WORD_CH}])"
)
_ALLCAPS_RE = re.compile(rf"(?<![{_WORD_CH}])([A-Z\u00C0-\u00DE][A-Z\u00C0-\u00DE0-9]{{1,14}})(?![{_WORD_CH}])")
_MID_CAP_RE = re.compile(rf"(?:[a-z0-9\u00E0-\u024F,;)\]]\s|['’])([A-Z\u00C0-\u00DE][a-z\u00E0-\u024F]{{2,}})(?![{_WORD_CH}])")
_TOKEN_RE = re.compile(rf"[{_WORD_CH}]+")
_PLACEHOLDER_RE = re.compile(r"#?\[(?:ENT|TAG|USER)_\d+\]|\[(?:ENTITY|HASHTAG|USER|URL|YEAR)\]")


def normalize_surface_key(text: str) -> str:
    """Unicode NFKD accent-folded lowercase key preserving alphanumeric boundaries."""
    cleaned = unicodedata.normalize("NFKD", text.strip())
    stripped = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
    stripped = stripped.lower().lstrip("#@")
    stripped = re.sub(r"[^a-z0-9]+", " ", stripped).strip()
    return stripped


def _build_accent_pattern(phrase: str) -> str:
    """Compile case/accent-tolerant regex body for an entity/hashtag phrase."""
    parts: List[str] = []
    i = 0
    s = phrase.strip()
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            parts.append(r"\s+")
            while i < len(s) and s[i].isspace():
                i += 1
            continue
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c)).lower()
        if base in _ACCENT_EQUIV:
            parts.append(_ACCENT_EQUIV[base])
        else:
            parts.append(re.escape(ch))
        i += 1
    return "".join(parts)


class _MappingTextView:
    """Lightweight read-only view supporting `ds.texts[idx]` without copying whole corpus."""

    def __init__(self, texts_by_idx: Dict[int, str], fallback: Sequence[str]) -> None:
        self._by_idx = texts_by_idx
        self._fallback = fallback

    def __getitem__(self, idx: int) -> str:
        return self._by_idx.get(idx, self._fallback[idx])


@dataclass
class _DatasetPromptProxy:
    name: str
    texts: _MappingTextView
    dates: Sequence[str]


def _trim_preserving_placeholders(text: str, max_chars: int = 140) -> str:
    """Trim anonymized text to `<= max_chars` without slicing `[ENT_k]` tokens in half."""
    single = text.replace("\n", " ").strip()
    if len(single) <= max_chars:
        return single
    cut = single[:max_chars]
    last_open = cut.rfind("[")
    last_close = cut.rfind("]")
    if last_open > last_close:
        cut = cut[:last_open].rstrip()
    else:
        # Also trim trailing dangling partial word if mid-token
        sp = cut.rfind(" ")
        if sp > max_chars // 2 and single[max_chars : max_chars + 1].isalnum():
            cut = cut[:sp].rstrip()
    return cut


class QueryAnonymizer:
    """Deterministic query-scoped entity/hashtag/proper-noun anonymizer."""

    def __init__(self, mode: str = "pseudonym") -> None:
        if mode not in {"pseudonym", "blind"}:
            raise ValueError(f"Unsupported anonymization mode: {mode!r}")
        self.mode = mode

    @staticmethod
    def _shown_cluster_ids_and_medoids(q: QueryItem) -> Tuple[List[int], List[int]]:
        if q.query_type == "merge_pair":
            cids = [c for c in [q.merge_c1, q.merge_c2] if c is not None]
            meds = list(q.merge_c1_medoids[:3]) + list(q.merge_c2_medoids[:3])
        elif q.query_type == "seam":
            cids = [q.seam_cluster_id] if q.seam_cluster_id is not None else []
            meds = list(q.seam_sub1_medoids[:3]) + list(q.seam_sub2_medoids[:3])
        elif q.query_type == "star_merge":
            cids = list(q.candidate_clusters)
            meds = list(q.primary_nodes)
        else:
            cids = list(q.candidate_clusters)
            meds = list(q.primary_nodes)
        return cids, meds

    def build_query_anonymized_inputs(
        self,
        q: QueryItem,
        ds: SocialDataset,
        summaries: Dict[int, ClusterSummary],
    ) -> Tuple[Dict[int, str], Dict[int, ClusterSummary], Dict[str, Any]]:
        """Return anonymized medoid texts, anonymized cluster summaries, and leakage audit."""
        cids, shown_medoids = self._shown_cluster_ids_and_medoids(q)
        medoid_set: Set[int] = set(shown_medoids)
        if q.query_type not in {"merge_pair", "seam", "star_merge"}:
            medoid_set = set(q.primary_nodes)
            for cid in cids:
                if cid in summaries:
                    medoid_set.update(summaries[cid].medoids[:3])

        raw_snippets: Dict[int, str] = {}
        for m in sorted(medoid_set):
            raw_snippets[m] = ds.texts[m].replace("\n", " ").strip()

        # Collect dataset entities & hashtags ONLY from shown medoids + shown top_keywords
        # (never pulling noisy unrelated tokens from hundreds of unseen cluster members).
        entity_surfaces: Set[str] = set()
        hashtag_surfaces: Set[str] = set()
        shown_entity_surfaces: Set[str] = set()
        shown_hashtag_surfaces: Set[str] = set()

        for m in sorted(medoid_set):
            for e in ds.entities[m]:
                es = str(e).strip()
                if len(es) >= 2:
                    entity_surfaces.add(es)
                    shown_entity_surfaces.add(es)
            for h in ds.hashtags[m]:
                hs = str(h).strip().lstrip("#")
                if len(hs) >= 2:
                    hashtag_surfaces.add(hs)
                    shown_hashtag_surfaces.add(hs)

        kw_tokens: List[str] = []
        if q.query_type != "seam":
            for cid in cids:
                if cid in summaries:
                    kw_tokens.extend(summaries[cid].top_keywords[:5])

        if kw_tokens:
            kw_norms = {normalize_surface_key(kw): kw for kw in kw_tokens if normalize_surface_key(kw)}
            for cid in cids:
                if cid not in summaries:
                    continue
                for m in summaries[cid].members:
                    for e in ds.entities[m]:
                        ek = normalize_surface_key(str(e))
                        if ek in kw_norms:
                            entity_surfaces.add(str(e).strip())
                            shown_entity_surfaces.add(str(e).strip())
                    for h in ds.hashtags[m]:
                        hk = normalize_surface_key(str(h))
                        if hk in kw_norms:
                            hs = str(h).strip().lstrip("#")
                            hashtag_surfaces.add(hs)
                            shown_hashtag_surfaces.add(hs)

        # Also extract inline hashtags, URLs, @mentions, years, and genuine missed proper nouns
        shown_texts_all: List[str] = [raw_snippets[m][:180] for m in sorted(medoid_set)] + kw_tokens
        extra_proper: Set[str] = set()
        inline_mentions: Set[str] = set()
        # Temporarily mask known NER entities before running Proper-Noun heuristic so
        # headline chains cannot swallow/override real dataset entities.
        ner_mask_pats = [
            re.compile(rf"(?<![{_WORD_CH}]){_build_accent_pattern(es)}(?![{_WORD_CH}])", re.IGNORECASE)
            for es in sorted(entity_surfaces, key=lambda s: -len(s))
            if normalize_surface_key(es) not in _AUDIT_STOPWORDS
        ]
        for txt in shown_texts_all:
            cleaned_tmp = _URL_RE.sub(" ", txt)
            for hm in _HASHTAG_RE.finditer(cleaned_tmp):
                tag_body = hm.group(0).lstrip("#")
                if len(tag_body) >= 2:
                    hashtag_surfaces.add(tag_body)
                    shown_hashtag_surfaces.add(tag_body)
            for mm in _MENTION_RE.finditer(cleaned_tmp):
                inline_mentions.add(mm.group(0).lstrip("@"))
            cleaned_no_handles = _HASHTAG_RE.sub(" ", _MENTION_RE.sub(" ", cleaned_tmp))
            for npat in ner_mask_pats:
                cleaned_no_handles = npat.sub(" ", cleaned_no_handles)
            for pm in _MULTI_CAP_RE.finditer(cleaned_no_handles):
                parts = pm.group(1).strip().split()
                if all(p.lower() not in _AUDIT_STOPWORDS and p.lower() not in _CAP_FUNCTION_WORDS for p in parts):
                    cand = " ".join(parts)
                    if len(cand) >= 4:
                        extra_proper.add(cand)
            for am in _ALLCAPS_RE.finditer(cleaned_no_handles):
                tok = am.group(1)
                if tok not in _UPPER_BOILERPLATE and tok.lower() not in _AUDIT_STOPWORDS and len(tok) >= 2:
                    extra_proper.add(tok)
            for cm in _MID_CAP_RE.finditer(cleaned_no_handles):
                tok = cm.group(1)
                if tok.lower() not in _AUDIT_STOPWORDS and len(tok) >= 3:
                    extra_proper.add(tok)

        # Build canonical key -> placeholder map: dataset NER entities first, then heuristics, then hashtags
        all_ent_keys: List[Tuple[str, str]] = []
        seen_keys: Set[str] = set()

        def _add_candidate(surf: str, kind: str) -> None:
            key = normalize_surface_key(surf)
            if not key or len(key) < 2 or key in _AUDIT_STOPWORDS:
                return
            if key not in seen_keys:
                seen_keys.add(key)
                all_ent_keys.append((key, kind))

        for surf in sorted(entity_surfaces, key=lambda s: (-len(s), normalize_surface_key(s))):
            _add_candidate(surf, "ENT")
        for surf in sorted(extra_proper, key=lambda s: (-len(s), normalize_surface_key(s))):
            _add_candidate(surf, "ENT")
        for surf in sorted(hashtag_surfaces, key=lambda s: (-len(s), normalize_surface_key(s))):
            _add_candidate(surf, "TAG")

        # Share same [ENT_k] ID across multi-word constituent tokens (>=5 chars, e.g., Lance & Armstrong -> [ENT_1])
        ent_id_by_key: Dict[str, str] = {}
        tag_id_by_key: Dict[str, str] = {}
        alias_to_placeholder: Dict[str, str] = {}

        if self.mode == "pseudonym":
            ent_idx = 0
            tag_idx = 0
            for key, kind in all_ent_keys:
                if kind == "ENT":
                    if key in alias_to_placeholder:
                        ent_id_by_key[key] = alias_to_placeholder[key]
                        continue
                    ent_idx += 1
                    ph = f"[ENT_{ent_idx}]"
                    ent_id_by_key[key] = ph
                    alias_to_placeholder[key] = ph
                    toks = key.split()
                    if len(toks) >= 2:
                        for part_tok in toks:
                            if (
                                len(part_tok) >= 5
                                and part_tok not in _AUDIT_STOPWORDS
                                and part_tok not in alias_to_placeholder
                            ):
                                alias_to_placeholder[part_tok] = ph
                else:
                    if key in ent_id_by_key or key in alias_to_placeholder:
                        base_ph = ent_id_by_key.get(key) or alias_to_placeholder[key]
                        tag_id_by_key[key] = "#" + base_ph
                    else:
                        tag_idx += 1
                        tag_id_by_key[key] = f"#[TAG_{tag_idx}]"
        else:
            for key, kind in all_ent_keys:
                if kind == "ENT":
                    ent_id_by_key[key] = "[ENTITY]"
                    alias_to_placeholder[key] = "[ENTITY]"
                else:
                    tag_id_by_key[key] = "[HASHTAG]"

        # Link mentions deterministically
        mention_map: Dict[str, str] = {}
        for idx_m, m_surf in enumerate(sorted(inline_mentions, key=lambda s: normalize_surface_key(s)), start=1):
            mk = normalize_surface_key(m_surf)
            mention_map[mk] = f"[USER_{idx_m}]" if self.mode == "pseudonym" else "[USER]"

        # Compile ONE fast single-pass alternation regex sorted longest-first
        raw_replacements: Dict[str, str] = {}
        for h in hashtag_surfaces:
            hk = normalize_surface_key(h)
            raw_replacements["#" + h] = tag_id_by_key.get(hk, "[HASHTAG]")
        for surf in entity_surfaces | extra_proper:
            sk = normalize_surface_key(surf)
            ph = ent_id_by_key.get(sk) or alias_to_placeholder.get(sk)
            if ph:
                raw_replacements[surf] = ph
        for h in hashtag_surfaces:
            hk = normalize_surface_key(h)
            if h not in raw_replacements and hk not in alias_to_placeholder and hk not in _AUDIT_STOPWORDS:
                raw_replacements[h] = tag_id_by_key.get(hk, "[HASHTAG]").lstrip("#")

        norm_hash_lookup = {normalize_surface_key(k): v for k, v in raw_replacements.items() if k.startswith("#")}
        norm_text_lookup = {normalize_surface_key(k): v for k, v in raw_replacements.items() if not k.startswith("#")}
        sorted_surfs = sorted(raw_replacements.keys(), key=lambda s: (-len(s), s.lower()))
        combined_pat: Optional[re.Pattern[str]] = None
        if sorted_surfs:
            alts = "|".join([r"\[(?:URL|YEAR|USER(?:_\d+)?)\]"] + [_build_accent_pattern(s) for s in sorted_surfs])
            combined_pat = re.compile(rf"(?<![{_WORD_CH}])({alts})(?:['’]s)?(?![{_WORD_CH}])", re.IGNORECASE)

        def _lookup_match(m: re.Match[str]) -> str:
            hit = m.group(1)
            if hit.startswith("["):
                return hit
            hk = normalize_surface_key(hit)
            if hit.startswith("#"):
                return norm_hash_lookup.get(hk, tag_id_by_key.get(hk, "[HASHTAG]"))
            return norm_text_lookup.get(hk, ent_id_by_key.get(hk) or alias_to_placeholder.get(hk) or "[ENTITY]")

        def _anonymize_str(src: str, is_keyword: bool = False) -> str:
            if not src:
                return src
            out = _URL_RE.sub("[URL]", src)
            out = _MENTION_RE.sub(
                lambda m: mention_map.get(normalize_surface_key(m.group(0)), "[USER]" if self.mode == "blind" else "[USER_0]"),
                out,
            )
            out = _YEAR_RE.sub("[YEAR]", out)
            if is_keyword:
                kn = normalize_surface_key(out)
                if out.startswith("#") and kn in tag_id_by_key:
                    return tag_id_by_key[kn]
                if kn in ent_id_by_key:
                    return ent_id_by_key[kn]
                if kn in alias_to_placeholder:
                    return alias_to_placeholder[kn]
                if kn in tag_id_by_key:
                    return tag_id_by_key[kn].lstrip("#")
            if combined_pat is not None:
                out = combined_pat.sub(_lookup_match, out)
            out = _HASHTAG_RE.sub(
                lambda m: m.group(0) if m.group(0).startswith("#[") else tag_id_by_key.get(normalize_surface_key(m.group(0)), "[HASHTAG]"),
                out,
            )
            return out

        anon_texts: Dict[int, str] = {}
        raw_shown_head: Dict[int, str] = {}
        for m, full_s in raw_snippets.items():
            raw_head = full_s[:140]
            raw_shown_head[m] = raw_head
            # Truncate to original 140-char info boundary before anonymizing (extending cleanly
            # only if char 140 splits a known entity/hashtag token mid-word).
            prefix = raw_head
            if len(full_s) > 140 and full_s[139:140].isalnum() and full_s[140:141].isalnum():
                tail_ext = full_s[140:165]
                sp = tail_ext.find(" ")
                ext_tok = tail_ext if sp < 0 else tail_ext[:sp]
                candidate = prefix + ext_tok
                last_word = _TOKEN_RE.findall(candidate)[-1] if _TOKEN_RE.findall(candidate) else ""
                if normalize_surface_key(last_word) in seen_keys or normalize_surface_key(last_word) in alias_to_placeholder:
                    prefix = candidate
                else:
                    # Drop clipped trailing fragment so no partial proper name slips through
                    last_sp = prefix.rfind(" ")
                    if last_sp > 80:
                        prefix = prefix[:last_sp].rstrip()
            anon_head = _anonymize_str(prefix, is_keyword=False)
            anon_texts[m] = _trim_preserving_placeholders(anon_head, max_chars=140)

        anon_summaries: Dict[int, ClusterSummary] = {}
        for cid in cids:
            if cid in summaries:
                s = summaries[cid]
                new_kw = [_anonymize_str(kw, is_keyword=True) for kw in s.top_keywords[:5]]
                anon_summaries[cid] = replace(s, top_keywords=new_kw)

        # Build ground-truth raw unmasked text vs anonymized text for audit
        raw_shown_blob = " \n ".join(
            [raw_shown_head[m] for m in sorted(medoid_set)] + kw_tokens
        )
        anon_shown_blob = " \n ".join(
            [anon_texts[m] for m in sorted(medoid_set)]
            + [kw for cid in cids if cid in anon_summaries for kw in anon_summaries[cid].top_keywords]
        )

        # Audit entity, hashtag, year, mention, URL masking strictly against raw shown blob
        masked_originals: Set[str] = set()
        ent_total = ent_masked = 0
        # Deduplicate shown entity strings by normalized key of len >= 3
        audit_ents: Dict[str, str] = {}
        for surf in shown_entity_surfaces:
            kn = normalize_surface_key(surf)
            if len(kn) >= 3 and kn not in _AUDIT_STOPWORDS and surf not in audit_ents.values():
                audit_ents[kn] = surf
        for kn, surf in audit_ents.items():
            pat = re.compile(rf"(?<![A-Za-z0-9_À-ÖØ-öø-ÿ]){_build_accent_pattern(surf)}(?![A-Za-z0-9_À-ÖØ-öø-ÿ])", re.IGNORECASE)
            if pat.search(raw_shown_blob):
                ent_total += 1
                if not pat.search(anon_shown_blob):
                    ent_masked += 1
                    masked_originals.add(kn)

        raw_tags = {m.group(0).lstrip("#").lower() for m in _HASHTAG_RE.finditer(raw_shown_blob) if len(m.group(0)) >= 3}
        # Only raw un-replaced hashtags (exclude #[TAG_k] / #[ENT_k])
        anon_tags = {
            m.group(0).lstrip("#").lower()
            for m in _HASHTAG_RE.finditer(_PLACEHOLDER_RE.sub(" ", anon_shown_blob))
            if len(m.group(0)) >= 3
        }
        tag_total = len(raw_tags)
        tag_masked = len(raw_tags - anon_tags)
        for t in raw_tags:
            kn = normalize_surface_key(t)
            if len(kn) >= 4 and kn not in _AUDIT_STOPWORDS:
                masked_originals.add(kn)

        raw_years = _YEAR_RE.findall(raw_shown_blob)
        anon_years = _YEAR_RE.findall(anon_shown_blob)
        year_total = len(raw_years)
        year_masked = max(0, year_total - len(anon_years))

        raw_urls = _URL_RE.findall(raw_shown_blob)
        anon_urls = _URL_RE.findall(anon_shown_blob)
        url_total = len(raw_urls)
        url_masked = max(0, url_total - len(anon_urls))

        raw_toks = [t for t in _TOKEN_RE.findall(raw_shown_blob) if len(t) >= 2]
        n_ph = len(_PLACEHOLDER_RE.findall(anon_shown_blob))

        audit = {
            "query_id": q.query_id,
            "entity_total": ent_total,
            "entity_masked": ent_masked,
            "hashtag_total": tag_total,
            "hashtag_masked": tag_masked,
            "year_total": year_total,
            "year_masked": year_masked,
            "url_total": url_total,
            "url_masked": url_masked,
            "raw_tokens": len(raw_toks),
            "placeholders_inserted": n_ph,
            "masked_originals": sorted(masked_originals),
        }
        return anon_texts, anon_summaries, audit


def audit_pre_anonymized_query(
    query_id: str,
    shown_medoids: Sequence[int],
    shown_keywords: Sequence[str],
    orig_ds: SocialDataset,
    anon_ds: SocialDataset,
) -> Dict[str, Any]:
    """Audit masking coverage of an upstream pre-anonymized dataset against original ground-truth metadata."""
    medoid_ids = sorted({int(m) for m in shown_medoids})
    raw_heads = [orig_ds.texts[m].replace("\n", " ").strip()[:140] for m in medoid_ids]
    anon_heads = [anon_ds.texts[m].replace("\n", " ").strip()[:200] for m in medoid_ids]
    raw_shown_blob = " \n ".join(raw_heads)
    anon_shown_blob = " \n ".join(anon_heads + [str(k) for k in shown_keywords if k])

    shown_entity_surfaces: Set[str] = set()
    for m in medoid_ids:
        for e in orig_ds.entities[m]:
            es = str(e).strip()
            if len(es) >= 2:
                shown_entity_surfaces.add(es)

    masked_originals: Set[str] = set()
    ent_total = ent_masked = 0
    audit_ents: Dict[str, str] = {}
    for surf in shown_entity_surfaces:
        kn = normalize_surface_key(surf)
        if len(kn) >= 3 and kn not in _AUDIT_STOPWORDS and surf not in audit_ents.values():
            audit_ents[kn] = surf
    for kn, surf in audit_ents.items():
        pat = re.compile(rf"(?<![{_WORD_CH}]){_build_accent_pattern(surf)}(?![{_WORD_CH}])", re.IGNORECASE)
        if pat.search(raw_shown_blob):
            ent_total += 1
            if not pat.search(anon_shown_blob):
                ent_masked += 1
                masked_originals.add(kn)

    raw_tags = {m.group(0).lstrip("#").lower() for m in _HASHTAG_RE.finditer(raw_shown_blob) if len(m.group(0)) >= 3}
    anon_tags = {
        m.group(0).lstrip("#").lower()
        for m in _HASHTAG_RE.finditer(_PLACEHOLDER_RE.sub(" ", anon_shown_blob))
        if len(m.group(0)) >= 3
    }
    tag_total = len(raw_tags)
    tag_masked = len(raw_tags - anon_tags)
    for t in raw_tags:
        kn = normalize_surface_key(t)
        if len(kn) >= 4 and kn not in _AUDIT_STOPWORDS:
            masked_originals.add(kn)

    raw_years = _YEAR_RE.findall(raw_shown_blob)
    anon_years = _YEAR_RE.findall(anon_shown_blob)
    year_total = len(raw_years)
    year_masked = max(0, year_total - len(anon_years))

    raw_urls = _URL_RE.findall(raw_shown_blob)
    anon_urls = _URL_RE.findall(anon_shown_blob)
    url_total = len(raw_urls)
    url_masked = max(0, url_total - len(anon_urls))

    raw_toks = [t for t in _TOKEN_RE.findall(raw_shown_blob) if len(t) >= 2]
    n_ph = len(_PLACEHOLDER_RE.findall(anon_shown_blob))

    return {
        "query_id": query_id,
        "entity_total": ent_total,
        "entity_masked": ent_masked,
        "hashtag_total": tag_total,
        "hashtag_masked": tag_masked,
        "year_total": year_total,
        "year_masked": year_masked,
        "url_total": url_total,
        "url_masked": url_masked,
        "raw_tokens": len(raw_toks),
        "placeholders_inserted": n_ph,
        "masked_originals": sorted(masked_originals),
    }


def check_summary_reconstruction_leakage(
    event_a: str,
    event_b: str,
    masked_originals: Sequence[str],
) -> List[str]:
    """Detect whether LLM summaries hallucinated/reconstructed masked entities from pretraining."""
    text = normalize_surface_key(f"{event_a} {event_b}")
    if not text:
        return []
    toks = set(text.split())
    leaked: List[str] = []
    for orig in masked_originals:
        kn = normalize_surface_key(orig)
        if len(kn) < 4 or kn in _AUDIT_STOPWORDS:
            continue
        if " " in kn:
            if f" {kn} " in f" {text} ":
                leaked.append(kn)
        else:
            if kn in toks:
                leaked.append(kn)
    return sorted(set(leaked))


class AnonymizationAuditStore:
    """Thread-safe collector for per-query masking coverage & example pairs."""

    def __init__(self, max_examples: int = 8) -> None:
        self._lock = threading.Lock()
        self.records: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.examples: Dict[Tuple[str, str], Dict[str, str]] = {}
        self._max_examples = max_examples

    def record(
        self,
        ds_name: str,
        q_id: str,
        audit: Dict[str, Any],
        raw_user_msg: str,
        anon_user_msg: str,
    ) -> None:
        key = (ds_name, q_id)
        with self._lock:
            self.records[key] = audit
            if len(self.examples) < self._max_examples and audit.get("entity_total", 0) >= 1:
                self.examples[key] = {
                    "dataset": ds_name,
                    "query_id": q_id,
                    "raw_user_msg": raw_user_msg,
                    "anon_user_msg": anon_user_msg,
                }

    def summarize_for_dataset(self, ds_prefix: str, executed_keys: Optional[Iterable[Tuple[str, str]]] = None) -> Dict[str, Any]:
        with self._lock:
            keys = (
                [k for k in executed_keys if k in self.records and k[0].startswith(ds_prefix)]
                if executed_keys is not None
                else [k for k in self.records if k[0].startswith(ds_prefix)]
            )
            recs = [self.records[k] for k in keys]
        ent_t = sum(r["entity_total"] for r in recs)
        ent_m = sum(r["entity_masked"] for r in recs)
        tag_t = sum(r["hashtag_total"] for r in recs)
        tag_m = sum(r["hashtag_masked"] for r in recs)
        yr_t = sum(r["year_total"] for r in recs)
        yr_m = sum(r["year_masked"] for r in recs)
        url_t = sum(r["url_total"] for r in recs)
        url_m = sum(r["url_masked"] for r in recs)
        raw_tok = sum(r["raw_tokens"] for r in recs)
        n_ph = sum(r["placeholders_inserted"] for r in recs)
        return {
            "n_queries": len(recs),
            "entity_total": ent_t,
            "entity_masked": ent_m,
            "entity_mask_rate": (ent_m / ent_t) if ent_t > 0 else 1.0,
            "hashtag_total": tag_t,
            "hashtag_masked": tag_m,
            "hashtag_mask_rate": (tag_m / tag_t) if tag_t > 0 else 1.0,
            "year_total": yr_t,
            "year_masked": yr_m,
            "year_mask_rate": (yr_m / yr_t) if yr_t > 0 else 1.0,
            "url_total": url_t,
            "url_masked": url_m,
            "url_mask_rate": (url_m / url_t) if url_t > 0 else 1.0,
            "raw_tokens": raw_tok,
            "placeholders_inserted": n_ph,
            "placeholder_density": (n_ph / raw_tok) if raw_tok > 0 else 0.0,
        }


def make_anonymized_prompt_fn(
    anon_mode: str = "pseudonym",
    audit_store: Optional[AnonymizationAuditStore] = None,
) -> Callable[[QueryItem, SocialDataset, Dict[int, ClusterSummary]], Tuple[str, str]]:
    """Create a prompt builder wrapping cluster/pairwise queries with entity anonymization."""
    anonymizer = QueryAnonymizer(mode=anon_mode)
    note = (
        "\nNote: Proper names, hashtags, mentions, URLs, and years have been anonymized with query-consistent placeholders "
        "([ENT_1], #[ENT_1], #[TAG_1], [USER_1], [URL], [YEAR]). Decide strictly from event structure and shared/disjoint placeholders."
        if anon_mode == "pseudonym"
        else "\nNote: Proper names, hashtags, mentions, URLs, and years have been blindly redacted without identity indices. "
        "Decide strictly from remaining predicates and relational context."
    )

    def _raw_pair(q: QueryItem, pds: Any, psumm: Dict[int, ClusterSummary]) -> Tuple[str, str]:
        if q.query_type == "merge_pair" and q.merge_c1_medoids and q.merge_c2_medoids:
            t1 = " | ".join(pds.texts[m][:130] for m in q.merge_c1_medoids[:2])
            t2 = " | ".join(pds.texts[m][:130] for m in q.merge_c2_medoids[:2])
            return build_pairwise_same_event_prompt(t1, t2)
        if q.query_type == "seam" and q.seam_sub1_medoids and q.seam_sub2_medoids:
            t1 = " | ".join(pds.texts[m][:130] for m in q.seam_sub1_medoids[:3])
            t2 = " | ".join(pds.texts[m][:130] for m in q.seam_sub2_medoids[:3])
            return build_pairwise_same_event_prompt(t1, t2)
        return build_post_assign_prompt(q.primary_nodes[0], q.candidate_clusters, psumm, pds)

    def _prompt_fn(
        q: QueryItem,
        ds: SocialDataset,
        summaries: Dict[int, ClusterSummary],
    ) -> Tuple[str, str]:
        _, raw_user = _raw_pair(q, ds, summaries)
        anon_texts, anon_summaries, audit = anonymizer.build_query_anonymized_inputs(q, ds, summaries)
        proxy_ds = _DatasetPromptProxy(
            name=ds.name,
            texts=_MappingTextView(anon_texts, ds.texts),
            dates=["[DATE]"] * len(ds.dates),
        )
        sys_msg, anon_user = _raw_pair(q, proxy_ds, anon_summaries)
        if audit_store is not None:
            audit_store.record(ds.name, q.query_id, audit, raw_user, anon_user)
        return sys_msg + note, anon_user

    return _prompt_fn
