import { useEffect, useMemo, useState } from "react";
import { BarChart3, Download, FileImage, FileSpreadsheet, Gauge, Layers3, Play, UploadCloud } from "lucide-react";
import { exportCsvUrl, exportExcelUrl, getJob, getResults, startProcessing, uploadReference, uploadStudents } from "./lib/api";
import type { Job, ReferenceDrawing, Result, Submission } from "./types";

const drawingTypes = ["orthographic", "isometric", "sectional"];

export function App() {
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [studentFiles, setStudentFiles] = useState<File[]>([]);
  const [drawingType, setDrawingType] = useState("orthographic");
  const [reference, setReference] = useState<ReferenceDrawing | null>(null);
  const [submissions, setSubmissions] = useState<Submission[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [results, setResults] = useState<Result[]>([]);
  const [selected, setSelected] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("Ready for a reference drawing.");

  const averageScore = useMemo(() => {
    if (!results.length) return 0;
    return results.reduce((sum, result) => sum + result.score, 0) / results.length;
  }, [results]);

  useEffect(() => {
    if (!job || ["completed", "failed"].includes(job.status)) return;
    const timer = window.setInterval(async () => {
      const next = await getJob(job.id);
      setJob(next);
      setMessage(`Processing ${next.processed}/${next.total}`);
      if (next.status === "completed") {
        const fresh = await getResults(reference?.id);
        setResults(fresh);
        setSelected(fresh[0] ?? null);
        setMessage("Batch processing complete.");
      }
    }, 1400);
    return () => window.clearInterval(timer);
  }, [job, reference?.id]);

  async function handleUploadReference() {
    if (!referenceFile) return;
    setBusy(true);
    setMessage("Analyzing reference drawing...");
    try {
      const uploaded = await uploadReference(referenceFile, drawingType);
      setReference(uploaded);
      setSubmissions([]);
      setResults([]);
      setSelected(null);
      setJob(null);
      setMessage(`Reference #${uploaded.id} loaded with ${uploaded.drawing_type} evaluation.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Reference upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleUploadStudents() {
    if (!reference || studentFiles.length === 0) return;
    setBusy(true);
    setMessage(`Uploading ${studentFiles.length} student drawing(s)...`);
    try {
      const uploaded = await uploadStudents(reference.id, studentFiles);
      setSubmissions(uploaded);
      setResults([]);
      setSelected(null);
      setMessage(`${uploaded.length} submissions queued for evaluation.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Student upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleProcess() {
    if (!reference) return;
    setBusy(true);
    setMessage("Starting processing job...");
    try {
      let activeSubmissions = submissions.filter((submission) => submission.reference_id === reference.id);
      if (!activeSubmissions.length && studentFiles.length) {
        setMessage(`Uploading ${studentFiles.length} student drawing(s)...`);
        activeSubmissions = await uploadStudents(reference.id, studentFiles);
        setSubmissions(activeSubmissions);
      }
      if (!activeSubmissions.length) {
        throw new Error("Upload at least one student drawing before processing.");
      }
      const started = await startProcessing(reference.id);
      setJob({ id: started.job_id, reference_id: reference.id, status: started.status, processed: 0, total: started.total });
      setMessage(`Processing 0/${started.total}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not start processing.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Layers3 size={22} /></div>
          <div>
            <strong>AI Engineering Drawing Evaluation System</strong>
            <span>Professor dashboard</span>
          </div>
        </div>
        <nav>
          <a href="#upload">Upload</a>
          <a href="#results">Results</a>
          <a href="#review">Review</a>
        </nav>
        <div className="status-panel">
          <span>Status</span>
          <strong>{message}</strong>
          {job && <progress value={job.total ? job.processed / job.total : 0} max={1} />}
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p>Batch Evaluation</p>
            <h1>Evaluate engineering drawing sheets with geometric scoring.</h1>
          </div>
          <div className="actions">
            <a className="icon-button" href={exportCsvUrl} title="Export CSV"><Download size={18} /> CSV</a>
            <a className="icon-button" href={exportExcelUrl} title="Export Excel"><FileSpreadsheet size={18} /> Excel</a>
          </div>
        </header>

        <section id="upload" className="upload-grid">
          <div className="panel primary-panel">
            <div className="panel-title">
              <FileImage size={20} />
              <h2>Reference Drawing</h2>
            </div>
            <div className="segmented">
              {drawingTypes.map((type) => (
                <button key={type} className={type === drawingType ? "active" : ""} onClick={() => setDrawingType(type)}>
                  {type}
                </button>
              ))}
            </div>
            <label className="drop-zone">
              <UploadCloud size={24} />
              <span>{referenceFile ? referenceFile.name : "Choose JPG, PNG, or PDF reference"}</span>
              <input type="file" accept=".jpg,.jpeg,.png,.pdf" onChange={(event) => setReferenceFile(event.target.files?.[0] ?? null)} />
            </label>
            <button className="command" disabled={!referenceFile || busy} onClick={handleUploadReference}>
              Analyze Reference
            </button>
          </div>

          <div className="panel">
            <div className="panel-title">
              <BarChart3 size={20} />
              <h2>Student Batch</h2>
            </div>
            <label className="drop-zone">
              <UploadCloud size={24} />
              <span>{studentFiles.length ? `${studentFiles.length} files selected` : "Choose 100-1000 answer sheets"}</span>
              <input
                type="file"
                accept=".jpg,.jpeg,.png,.pdf"
                multiple
                onChange={(event) => setStudentFiles(Array.from(event.target.files ?? []))}
              />
            </label>
            <div className="button-row">
              <button className="command secondary" disabled={!reference || !studentFiles.length || busy} onClick={handleUploadStudents}>
                Upload Batch
              </button>
              <button className="command" disabled={!reference || busy || (!submissions.length && !studentFiles.length)} onClick={handleProcess}>
                <Play size={16} /> Process
              </button>
            </div>
          </div>
        </section>

        <section className="metrics">
          <Metric icon={<Gauge size={20} />} label="Average Marks" value={averageScore ? averageScore.toFixed(1) : "--"} />
          <Metric icon={<FileImage size={20} />} label="Submissions" value={String(submissions.length || results.length || 0)} />
          <Metric icon={<Layers3 size={20} />} label="Completed" value={String(results.length)} />
        </section>

        <section id="results" className="results-layout">
          <div className="table-wrap">
            <div className="section-heading">
              <h2>Batch Results</h2>
              <span>{results.length} evaluated</span>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Student</th>
                  <th>File</th>
                  <th>Marks</th>
                  <th>Weakest Area</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result) => {
                  const category = result.metrics.category_scores ?? {};
                  const weakest = Object.entries(category).sort((a, b) => a[1] - b[1])[0]?.[0] ?? "geometry";
                  return (
                    <tr key={result.id} className={selected?.id === result.id ? "selected" : ""} onClick={() => setSelected(result)}>
                      <td>{result.student_id}</td>
                      <td>{result.filename}</td>
                      <td><strong>{result.score.toFixed(1)}</strong></td>
                      <td>{weakest}</td>
                    </tr>
                  );
                })}
                {!results.length && (
                  <tr>
                    <td colSpan={4} className="empty">No evaluated submissions yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div id="review" className="review-panel">
            <div className="section-heading">
              <h2>Drawing Review</h2>
              <span>{selected ? selected.student_id : "None selected"}</span>
            </div>
            {selected ? (
              <>
                <div
                  className="score-ring"
                  style={{
                    background: `conic-gradient(#d3ff7e 0 ${(selected.percentage ?? selected.score / 100) * 100}%, #e8eee8 ${(selected.percentage ?? selected.score / 100) * 100}% 100%)`
                  }}
                >
                  {selected.score.toFixed(0)}
                </div>
                <p className="feedback">{selected.feedback}</p>
                <div className="category-grid">
                  {Object.entries(selected.metrics.category_scores ?? {}).map(([key, value]) => (
                    <div key={key}>
                      <span>{key}</span>
                      <strong>{value.toFixed(1)}</strong>
                    </div>
                  ))}
                </div>
                <ul className="errors">
                  {selected.errors.slice(0, 6).map((error) => <li key={error}>{error}</li>)}
                </ul>
                <div className="visuals">
                  {selected.overlay_url && <img src={selected.overlay_url} alt="Annotated overlay" />}
                  {selected.heatmap_url && <img src={selected.heatmap_url} alt="Deviation heatmap" />}
                </div>
              </>
            ) : (
              <p className="empty">Select a processed result to inspect overlays and feedback.</p>
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="metric">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
