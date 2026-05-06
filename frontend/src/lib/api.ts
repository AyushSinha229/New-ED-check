import type { Job, ReferenceDrawing, Result, Submission } from "../types";

const API_BASE = "/api";

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function uploadReference(file: File, drawingType: string): Promise<ReferenceDrawing> {
  const form = new FormData();
  form.append("file", file);
  form.append("drawing_type", drawingType);
  return parse<ReferenceDrawing>(await fetch(`${API_BASE}/upload-reference`, { method: "POST", body: form }));
}

export async function uploadStudents(referenceId: number, files: File[]): Promise<Submission[]> {
  const form = new FormData();
  form.append("reference_id", String(referenceId));
  files.forEach((file) => form.append("files", file));
  return parse<Submission[]>(await fetch(`${API_BASE}/upload-students`, { method: "POST", body: form }));
}

export async function startProcessing(referenceId: number): Promise<{ job_id: string; status: string; total: number }> {
  const form = new FormData();
  form.append("reference_id", String(referenceId));
  return parse(await fetch(`${API_BASE}/process`, { method: "POST", body: form }));
}

export async function getJob(jobId: string): Promise<Job> {
  return parse<Job>(await fetch(`${API_BASE}/jobs/${jobId}`));
}

export async function getResults(referenceId?: number): Promise<Result[]> {
  const suffix = referenceId ? `?reference_id=${referenceId}` : "";
  return parse<Result[]>(await fetch(`${API_BASE}/results${suffix}`));
}

export const exportCsvUrl = `${API_BASE}/export/csv`;
export const exportExcelUrl = `${API_BASE}/export/excel`;
