import { useState } from "react";
import { toast } from "sonner";

import {
  useCreateEducation,
  useCreateExperience,
  useCreateProject,
  useUpdateEducation,
  useUpdateExperience,
  useUpdateProject,
} from "@/hooks/queries";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

export type ResumeEditorKind = "experience" | "education" | "project";

function lines(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

function splitLines(value: string): string[] {
  return value.split("\n").map((item) => item.trim()).filter(Boolean);
}

function splitCommaList(value: string): string[] {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

export function ResumeItemEditor({
  kind,
  item,
  onClose,
}: {
  kind: ResumeEditorKind;
  item?: any;
  onClose: () => void;
}) {
  const editing = Boolean(item?.id);
  const createExperience = useCreateExperience();
  const updateExperience = useUpdateExperience();
  const createEducation = useCreateEducation();
  const updateEducation = useUpdateEducation();
  const createProject = useCreateProject();
  const updateProject = useUpdateProject();

  const [title, setTitle] = useState(item?.title || "");
  const [company, setCompany] = useState(item?.company || "");
  const [location, setLocation] = useState(item?.location || "");
  const [institution, setInstitution] = useState(item?.institution || item?.school || "");
  const [degree, setDegree] = useState(item?.degree || "");
  const [fieldOfStudy, setFieldOfStudy] = useState(item?.field_of_study || "");
  const [grade, setGrade] = useState(item?.grade || "");
  const [name, setName] = useState(item?.name || item?.title || "");
  const [description, setDescription] = useState(item?.description || "");
  const [link, setLink] = useState(item?.link || "");
  const [startDate, setStartDate] = useState(item?.start_date || "");
  const [endDate, setEndDate] = useState(item?.end_date || "");
  const [bullets, setBullets] = useState(lines(item?.description_bullets || item?.bullets).join("\n"));
  const [notes, setNotes] = useState(lines(item?.notes).join("\n"));
  const [technologies, setTechnologies] = useState(lines(item?.technologies || item?.tech).join(", "));

  const isPending =
    createExperience.isPending || updateExperience.isPending ||
    createEducation.isPending || updateEducation.isPending ||
    createProject.isPending || updateProject.isPending;

  const finish = (label: string) => ({ onSuccess: () => { toast.success(label); onClose(); }, onError: (err: any) => toast.error("Could not save", { description: err?.message }) });

  const save = () => {
    if (kind === "experience") {
      const fields = { title, company, location, start_date: startDate, end_date: endDate, description_bullets: splitLines(bullets), technologies: splitCommaList(technologies) };
      if (editing) updateExperience.mutate({ id: item.id, fields }, finish("Experience updated"));
      else createExperience.mutate(fields, finish("Experience added"));
      return;
    }
    if (kind === "education") {
      const fields = { institution, degree, field_of_study: fieldOfStudy, start_date: startDate, end_date: endDate, grade, notes: splitLines(notes) };
      if (editing) updateEducation.mutate({ id: item.id, fields }, finish("Education updated"));
      else createEducation.mutate(fields, finish("Education added"));
      return;
    }
    const fields = { name, description, link, technologies: splitCommaList(technologies) };
    if (editing) updateProject.mutate({ id: item.id, fields }, finish("Project updated"));
    else createProject.mutate(fields, finish("Project added"));
  };

  const heading = `${editing ? "Edit" : "Add"} ${kind}`;
  return (
    <>
      <DialogHeader><DialogTitle>{heading}</DialogTitle></DialogHeader>
      <div className="max-h-[65vh] space-y-3 overflow-y-auto py-1">
        {kind === "experience" && <>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Title"><Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Software Engineer" /></Field>
            <Field label="Company"><Input value={company} onChange={(e) => setCompany(e.target.value)} placeholder="Company name" /></Field>
          </div>
          <Field label="Location"><Input value={location} onChange={(e) => setLocation(e.target.value)} placeholder="Delhi, India" /></Field>
          <DateFields startDate={startDate} endDate={endDate} onStart={setStartDate} onEnd={setEndDate} />
          <Field label="Highlights (one per line)"><Textarea value={bullets} onChange={(e) => setBullets(e.target.value)} placeholder="Built a feature…\nImproved performance by…" /></Field>
        </>}
        {kind === "education" && <>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Institution"><Input value={institution} onChange={(e) => setInstitution(e.target.value)} placeholder="University name" /></Field>
            <Field label="Degree"><Input value={degree} onChange={(e) => setDegree(e.target.value)} placeholder="B.Tech" /></Field>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Field of study"><Input value={fieldOfStudy} onChange={(e) => setFieldOfStudy(e.target.value)} placeholder="Computer Science" /></Field>
            <Field label="Grade"><Input value={grade} onChange={(e) => setGrade(e.target.value)} placeholder="8.5 CGPA" /></Field>
          </div>
          <DateFields startDate={startDate} endDate={endDate} onStart={setStartDate} onEnd={setEndDate} />
          <Field label="Notes (one per line)"><Textarea value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Relevant coursework\nHonors" /></Field>
        </>}
        {kind === "project" && <>
          <Field label="Project name"><Input value={name} onChange={(e) => setName(e.target.value)} placeholder="CareerAtlas" /></Field>
          <Field label="Description"><Textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="What did you build and why?" /></Field>
          <Field label="Project URL"><Input value={link} onChange={(e) => setLink(e.target.value)} placeholder="https://github.com/…" /></Field>
        </>}
        {(kind === "experience" || kind === "project") && <Field label="Technologies (comma-separated)"><Input value={technologies} onChange={(e) => setTechnologies(e.target.value)} placeholder="React, TypeScript, PostgreSQL" /></Field>}
      </div>
      <DialogFooter>
        <Button variant="outline" onClick={onClose} disabled={isPending}>Cancel</Button>
        <Button onClick={save} disabled={isPending}>{isPending ? "Saving…" : "Save"}</Button>
      </DialogFooter>
    </>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="block"><span className="text-xs font-medium text-muted-foreground">{label}</span><div className="mt-1">{children}</div></label>;
}

function DateFields({ startDate, endDate, onStart, onEnd }: { startDate: string; endDate: string; onStart: (value: string) => void; onEnd: (value: string) => void }) {
  return <div className="grid grid-cols-2 gap-3"><Field label="Start date"><Input value={startDate} onChange={(e) => onStart(e.target.value)} placeholder="Jan 2024" /></Field><Field label="End date"><Input value={endDate} onChange={(e) => onEnd(e.target.value)} placeholder="Present" /></Field></div>;
}
