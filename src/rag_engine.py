import re


class SimpleRAGEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2", collection_name="asset_records"):
        self.model_name = model_name
        self.collection_name = collection_name
        self.model = None
        self.client = None
        self.collection = None
        self.chunks = []
        self.asset_ids = set()

    def build_index(self, assets, work_orders, inspections, failures):
        self._load_dependencies()
        self.chunks = self._build_chunks(assets, work_orders, inspections, failures)
        self.asset_ids = {str(asset_id) for asset_id in assets["asset_id"]}

        documents = [chunk["text"] for chunk in self.chunks]
        metadatas = [chunk["metadata"] for chunk in self.chunks]
        ids = [chunk["id"] for chunk in self.chunks]
        embeddings = self.model.encode(documents).tolist()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Industrial asset intelligence demo records"},
        )

        existing = self.collection.get()
        if existing.get("ids"):
            self.collection.delete(ids=existing["ids"])

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        return self

    def retrieve(self, question, top_k=5):
        if self.collection is None:
            raise RuntimeError("RAG index has not been built yet.")

        query_embedding = self.model.encode([question]).tolist()[0]
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        asset_id = self._asset_id_from_question(question)
        if asset_id:
            query_args["where"] = {"asset_id": asset_id}

        results = self.collection.query(**query_args)

        retrieved = []
        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            retrieved.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )
        return retrieved

    def answer(self, question, top_k=5):
        retrieved = self.retrieve(question, top_k=top_k)
        sources = self.format_sources(retrieved)
        answer = self.summarise(question, retrieved)
        return {
            "answer": answer,
            "sources": sources,
            "retrieved": retrieved,
        }

    def summarise(self, question, retrieved):
        if not retrieved:
            return "No relevant records were retrieved for this question."

        asset_ids = self._ordered_unique(
            item["metadata"]["asset_id"] for item in retrieved if item["metadata"].get("asset_id")
        )
        asset_text = self._select_asset_text(question, asset_ids)
        verb = "appear" if asset_text.startswith("assets ") else "appears"
        evidence_text = self._extract_evidence(retrieved)

        if "risk" in question.lower() or "urgent" in question.lower():
            return (
                f"Based on the retrieved asset records, {asset_text} {verb} high "
                f"risk or operationally important because the local index retrieved "
                f"records showing {evidence_text}. Review the sources and context "
                "below for the exact evidence."
            )

        return (
            f"Based on the retrieved asset records, {asset_text} {verb} relevant to "
            f"this question. The local index retrieved records showing {evidence_text}. "
            "Review the sources and context below for the exact records used."
        )

    def format_sources(self, retrieved):
        seen = set()
        sources = []
        for item in retrieved:
            metadata = item["metadata"]
            key = (metadata["source_file"], metadata["record_id"])
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "source_file": metadata["source_file"],
                    "record_id": metadata["record_id"],
                    "asset_id": metadata.get("asset_id", ""),
                }
            )
        return sources

    def _load_dependencies(self):
        if self.model is not None and self.client is not None:
            return

        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))

    def _build_chunks(self, assets, work_orders, inspections, failures):
        chunks = []

        for _, row in assets.iterrows():
            asset_id = row["asset_id"]
            chunks.append(
                self._chunk(
                    chunk_id=f"assets-{asset_id}",
                    source_file="assets.csv",
                    record_id=asset_id,
                    asset_id=asset_id,
                    text=(
                        f"Asset record {asset_id}: {row['asset_name']} is a "
                        f"{row['criticality']} criticality {row['asset_type']} in "
                        f"{row['plant_area']}, installed in {row['install_year']}."
                    ),
                )
            )

        for _, row in work_orders.iterrows():
            record_id = row["work_order_id"]
            completed_date = self._format_optional_date(row["completed_date"])
            completed_text = f", completed {completed_date}" if completed_date else ""
            chunks.append(
                self._chunk(
                    chunk_id=f"work-orders-{record_id}",
                    source_file="work_orders.csv",
                    record_id=record_id,
                    asset_id=row["asset_id"],
                    text=(
                        f"Work order {record_id} for asset {row['asset_id']}: "
                        f"{row['work_type']} work, {row['description']}. Status is "
                        f"{row['status']} with {row['priority']} priority. Planned "
                        f"{self._format_optional_date(row['planned_date'])}{completed_text}. "
                        f"Downtime was {int(row['downtime_hours'])} hours and cost was "
                        f"{row['cost']}."
                    ),
                )
            )

        for _, row in inspections.iterrows():
            record_id = row["inspection_id"]
            chunks.append(
                self._chunk(
                    chunk_id=f"inspections-{record_id}",
                    source_file="inspections.csv",
                    record_id=record_id,
                    asset_id=row["asset_id"],
                    text=(
                        f"Inspection {record_id} for asset {row['asset_id']} on "
                        f"{self._format_optional_date(row['inspection_date'])}: condition "
                        f"score {int(row['condition_score'])}. Findings: {row['findings']}. "
                        f"Recommendation: {row['recommendation']}."
                    ),
                )
            )

        for _, row in failures.iterrows():
            record_id = row["failure_id"]
            chunks.append(
                self._chunk(
                    chunk_id=f"failures-{record_id}",
                    source_file="failure_events.csv",
                    record_id=record_id,
                    asset_id=row["asset_id"],
                    text=(
                        f"Failure event {record_id} for asset {row['asset_id']} on "
                        f"{self._format_optional_date(row['failure_date'])}: failure mode "
                        f"was {row['failure_mode']}. Root cause was {row['root_cause']}. "
                        f"Downtime was {int(row['downtime_hours'])} hours."
                    ),
                )
            )

        return chunks

    def _chunk(self, chunk_id, source_file, record_id, asset_id, text):
        return {
            "id": chunk_id,
            "text": text,
            "metadata": {
                "source_file": source_file,
                "record_id": str(record_id),
                "asset_id": str(asset_id),
            },
        }

    def _describe_evidence(self, source_types):
        labels = {
            "assets": "asset register details",
            "work_orders": "maintenance work order history",
            "inspections": "inspection findings and condition scores",
            "failure_events": "failure history and downtime evidence",
        }
        evidence = [labels[source_type] for source_type in sorted(source_types) if source_type in labels]
        if not evidence:
            return "matching source records"
        if len(evidence) == 1:
            return evidence[0]
        return ", ".join(evidence[:-1]) + f", and {evidence[-1]}"

    def _extract_evidence(self, retrieved):
        text = " ".join(item["text"] for item in retrieved)
        text_lower = text.lower()
        evidence = []

        if "high criticality" in text_lower:
            evidence.append("high asset criticality")
        elif "medium criticality" in text_lower:
            evidence.append("medium asset criticality")

        condition_scores = [
            int(score) for score in re.findall(r"condition score (\d+)", text_lower)
        ]
        if condition_scores:
            evidence.append(f"inspection condition score {min(condition_scores)}")

        if "status is open" in text_lower and "critical priority" in text_lower:
            evidence.append("an open critical work order")
        elif "status is open" in text_lower and "high priority" in text_lower:
            evidence.append("an open high-priority work order")
        elif "status is open" in text_lower:
            evidence.append("an open work order")

        if "failure event" in text_lower:
            if "vibration" in text_lower:
                evidence.append("previous vibration-related failure history")
            else:
                evidence.append("previous failure history")

        if "downtime was" in text_lower:
            evidence.append("recorded downtime")

        if not evidence:
            source_types = {
                item["metadata"]["source_file"].replace(".csv", "")
                for item in retrieved
            }
            return self._describe_evidence(source_types)

        return self._join_phrases(self._ordered_unique(evidence))

    def _select_asset_text(self, question, asset_ids):
        question_lower = question.lower()
        for asset_id in asset_ids:
            if asset_id.lower() in question_lower:
                return asset_id
        if not asset_ids:
            return "the retrieved assets"
        if len(asset_ids) == 1:
            return asset_ids[0]
        return "assets " + ", ".join(asset_ids)

    def _join_phrases(self, phrases):
        if len(phrases) == 1:
            return phrases[0]
        return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    def _ordered_unique(self, values):
        seen = set()
        unique_values = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            unique_values.append(value)
        return unique_values

    def _format_optional_date(self, value):
        if value is None:
            return ""
        try:
            if value != value:
                return ""
        except TypeError:
            pass
        if hasattr(value, "date"):
            return str(value.date())
        return str(value)

    def _asset_id_from_question(self, question):
        question_lower = question.lower()
        for asset_id in self.asset_ids:
            if asset_id.lower() in question_lower:
                return asset_id
        return None
