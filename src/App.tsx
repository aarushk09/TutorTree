import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import type { Scenario } from "./data/humanRatingTask";
import { scenarios } from "./data/humanRatingTask";
import { isSupabaseConfigured, supabase } from "./supabaseClient";

type Selection = "A" | "B" | "Tie";
type Stage = "landing" | "tour" | "evaluation" | "feedback" | "finished";
type EvaluatorRole =
  | "University Faculty"
  | "K-12 Teacher"
  | "Student"
  | "EdTech Researcher"
  | "Other";

const SESSION_KEY = "neurips-human-eval-session-id";
const EVALUATOR_ROLE_KEY = "neurips-human-eval-evaluator-role";
const evaluatorRoles: EvaluatorRole[] = [
  "University Faculty",
  "K-12 Teacher",
  "Student",
  "EdTech Researcher",
  "Other",
];

const mockScenario: Scenario = {
  id: "mock_example",
  studentProfile: "First-Year Introductory Learner",
  chatContext:
    "student: I understand that a loop repeats code, but I do not know when it stops.\nteacher: What part of the loop tells the computer whether to keep going?\nstudent: I think it has something to do with the condition, but I am not sure how to check it.",
  interventionA:
    "hint: Focus on the condition first. Ask whether it is true before each repetition.",
  interventionB:
    "socratic_prompt: What would happen if the condition became false before the loop began?",
};

function getSessionId() {
  const existing = localStorage.getItem(SESSION_KEY);
  if (existing) return existing;

  const id = crypto.randomUUID ? crypto.randomUUID() : fallbackUuid();
  localStorage.setItem(SESSION_KEY, id);
  return id;
}

function fallbackUuid() {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex
    .slice(6, 8)
    .join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
}

function getStoredEvaluatorRole(): EvaluatorRole | "" {
  const storedRole = localStorage.getItem(EVALUATOR_ROLE_KEY);
  return evaluatorRoles.includes(storedRole as EvaluatorRole)
    ? (storedRole as EvaluatorRole)
    : "";
}

function Transcript({ value }: { value: string }) {
  const lines = value.split("\n").filter(Boolean);

  return (
    <div className="transcript" aria-label="Chat transcript">
      {lines.map((line, index) => {
        const [speaker, ...body] = line.split(":");
        return (
          <div className="turn" key={`${speaker}-${index}`}>
            <span className="speaker">{speaker}</span>
            <p>{body.join(":").trim()}</p>
          </div>
        );
      })}
    </div>
  );
}

function formatInterventionText(rawText: string) {
  return rawText.replace(/^[^:]+:\s+/, "");
}

type EvaluationFrameProps = {
  scenario: Scenario;
  selected: Selection | null;
  stageLabel: string;
  title: string;
  progressLabel: string;
  progress: number;
  isMock?: boolean;
  onSelect: (choice: Selection) => void;
};

function EvaluationFrame({
  scenario,
  selected,
  stageLabel,
  title,
  progressLabel,
  progress,
  isMock = false,
  onSelect,
}: EvaluationFrameProps) {
  return (
    <>
      <header className="topbar">
        <div>
          <p className="kicker">{stageLabel}</p>
          <h1>{title}</h1>
        </div>
        <div className="progress-block" aria-label="Evaluation progress">
          <span>{progressLabel}</span>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
        </div>
      </header>

      {isMock && (
        <aside className="mock-banner" role="status">
          MOCK EXAMPLE
        </aside>
      )}

      <section className="workspace">
        <article className="context-panel field-profile">
          {isMock && (
            <div className="callout callout-profile">Student profile</div>
          )}
          <div className="panel-heading">
            <span>Student Profile</span>
            <strong>{scenario.studentProfile}</strong>
          </div>
          <div className="field-transcript">
            {isMock && (
              <div className="callout callout-transcript">Chat transcript</div>
            )}
            <Transcript value={scenario.chatContext} />
          </div>
        </article>

        <section className="comparison field-comparison" aria-label="Intervention comparison">
          {isMock && (
            <div className="callout callout-comparison">
              Compare A and B without knowing model identity
            </div>
          )}
          <InterventionCard
            label="A"
            text={formatInterventionText(scenario.interventionA)}
            selected={selected === "A"}
            shortcut="Left Arrow"
            onSelect={() => onSelect("A")}
          />
          <InterventionCard
            label="B"
            text={formatInterventionText(scenario.interventionB)}
            selected={selected === "B"}
            shortcut="Right Arrow"
            onSelect={() => onSelect("B")}
          />
          <button
            type="button"
            className={`tie-choice-button ${selected === "Tie" ? "selected" : ""}`}
            onClick={() => onSelect("Tie")}
          >
            Mark Tie
          </button>
        </section>
      </section>
    </>
  );
}

function InterventionCard({
  label,
  text,
  selected,
  shortcut,
  onSelect,
}: {
  label: "A" | "B";
  text: string;
  selected: boolean;
  shortcut: string;
  onSelect: () => void;
}) {
  return (
    <article className={`intervention ${selected ? "selected" : ""}`}>
      <div className="intervention-label">Intervention {label}</div>
      <p>{text}</p>
      <button
        type="button"
        className="choice-button"
        aria-keyshortcuts={label === "A" ? "ArrowLeft" : "ArrowRight"}
        onClick={onSelect}
      >
        Select {label}
        <span>{shortcut}</span>
      </button>
    </article>
  );
}

export function App() {
  const [sessionId] = useState(getSessionId);
  const [stage, setStage] = useState<Stage>("landing");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [evaluatorRole, setEvaluatorRole] = useState<EvaluatorRole | "">(
    getStoredEvaluatorRole,
  );
  const [selected, setSelected] = useState<Selection | null>(null);
  const [reasoning, setReasoning] = useState("");
  const [feedbackText, setFeedbackText] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isAdvancing, setIsAdvancing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const currentScenario = scenarios[currentIndex];
  const progress = useMemo(
    () => Math.round(((currentIndex + 1) / scenarios.length) * 100),
    [currentIndex],
  );

  const updateEvaluatorRole = (role: EvaluatorRole) => {
    setEvaluatorRole(role);
    localStorage.setItem(EVALUATOR_ROLE_KEY, role);
  };

  useEffect(() => {
    if (selected && stage === "evaluation") {
      textareaRef.current?.focus();
    }
  }, [selected, stage]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (stage !== "evaluation") return;
      const target = event.target as HTMLElement | null;
      if (target?.tagName === "TEXTAREA" || target?.tagName === "INPUT") {
        return;
      }

      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setSelected("A");
        setError(null);
      }

      if (event.key === "ArrowRight") {
        event.preventDefault();
        setSelected("B");
        setError(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [stage]);

  const selectIntervention = (choice: Selection) => {
    setSelected(choice);
    setError(null);
  };

  const submitEvaluation = async (event: FormEvent) => {
    event.preventDefault();
    if (!currentScenario || !selected || isSubmitting) return;

    if (!isSupabaseConfigured || !supabase) {
      setError("Supabase is missing VITE_SUPABASE_ANON_KEY.");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    const { error: insertError } = await supabase
      .from("human_evaluations")
      .insert({
        session_id: sessionId,
        scenario_id: currentScenario.id,
        selected_intervention: selected,
        evaluator_role: evaluatorRole,
        reasoning: reasoning.trim() || null,
      });

    if (insertError) {
      setError(insertError.message);
      setIsSubmitting(false);
      return;
    }

    setIsAdvancing(true);
    window.setTimeout(() => {
      if (currentIndex + 1 >= scenarios.length) {
        setStage("feedback");
      } else {
        setCurrentIndex((index) => index + 1);
      }
      setSelected(null);
      setReasoning("");
      setIsSubmitting(false);
      setIsAdvancing(false);
    }, 180);
  };

  const submitFeedback = async (event: FormEvent) => {
    event.preventDefault();

    if (!isSupabaseConfigured || !supabase) {
      setError("Supabase is missing VITE_SUPABASE_ANON_KEY.");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    const { error: insertError } = await supabase.from("human_feedback").insert({
      session_id: sessionId,
      feedback_text: feedbackText.trim() || null,
    });

    if (insertError) {
      setError(insertError.message);
      setIsSubmitting(false);
      return;
    }

    setStage("finished");
    setIsSubmitting(false);
  };

  if (stage === "landing") {
    return (
      <main className="intro-screen">
        <section className="intro-copy">
          <p className="kicker">Human Preference Study</p>
          <h1>Welcome.</h1>
          <p>
            This study evaluates AI tutoring interventions for educational
            dialogue. You will be presented with 30 brief scenarios showing a
            student's profile and chat history. Your task is to select which AI
            intervention is pedagogically superior.
          </p>
          <div className="disclaimer-block">
            <p>
              All responses are recorded completely anonymously to ensure
              unbiased, honest feedback.
            </p>
            <p>
              This is a double-blind study; the AI models generating the
              responses are hidden and randomized.
            </p>
          </div>
          <fieldset className="role-selector">
            <legend>Evaluator Role</legend>
            <div className="role-options">
              {evaluatorRoles.map((role) => (
                <label key={role} className="role-option">
                  <input
                    type="radio"
                    name="evaluator-role"
                    value={role}
                    checked={evaluatorRole === role}
                    onChange={() => updateEvaluatorRole(role)}
                    required
                  />
                  <span>{role}</span>
                </label>
              ))}
            </div>
          </fieldset>
          <button
            type="button"
            className="primary-action"
            disabled={!evaluatorRole}
            onClick={() => setStage("tour")}
          >
            Start Tutorial
          </button>
        </section>
      </main>
    );
  }

  if (stage === "tour") {
    return (
      <main className="app-shell">
        <EvaluationFrame
          scenario={mockScenario}
          selected={selected}
          stageLabel="Tutorial"
          title="Mock Example"
          progressLabel="Practice only"
          progress={0}
          isMock
          onSelect={selectIntervention}
        />
        <section className="tour-footer">
          <div>
            <p className="kicker">Keyboard Shortcuts</p>
            <p>
              Press Left Arrow for A, Right Arrow for B. Use Tie only when the
              interventions are pedagogically equivalent.
            </p>
          </div>
          <button
            type="button"
            className="primary-action compact"
            onClick={() => {
              setSelected(null);
              setError(null);
              setStage("evaluation");
            }}
          >
            I understand, begin evaluation
          </button>
        </section>
      </main>
    );
  }

  if (stage === "feedback") {
    return (
      <main className="feedback-screen">
        <form className="feedback-form" onSubmit={submitFeedback}>
          <p className="kicker">Debrief</p>
          <h1>Evaluation Complete.</h1>
          <p>Thank you for your contribution to this research.</p>
          <label htmlFor="feedback">
            Do you have any general feedback on the study, the interventions,
            or the AI's pedagogical choices?
          </label>
          <textarea
            id="feedback"
            value={feedbackText}
            maxLength={8000}
            onChange={(event) => setFeedbackText(event.target.value)}
            placeholder="Optional feedback"
          />
          <button type="submit" className="primary-action" disabled={isSubmitting}>
            {isSubmitting ? "Submitting" : "Submit Feedback & Finish"}
          </button>
          {error && <p className="error-message standalone">{error}</p>}
        </form>
      </main>
    );
  }

  if (stage === "finished") {
    return (
      <main className="complete-screen">
        <p className="kicker">Human Evaluation</p>
        <h1>Finished.</h1>
        <p>Your responses and feedback have been recorded.</p>
      </main>
    );
  }

  return (
    <main className={`app-shell ${isAdvancing ? "is-advancing" : ""}`}>
      <EvaluationFrame
        scenario={currentScenario}
        selected={selected}
        stageLabel="Blind A/B Tutoring Evaluation"
        title={`Scenario ${currentIndex + 1} of ${scenarios.length}`}
        progressLabel={`${progress}% complete`}
        progress={progress}
        onSelect={selectIntervention}
      />

      {!isSupabaseConfigured && (
        <aside className="system-notice" role="status">
          Add <code>VITE_SUPABASE_ANON_KEY</code> to your local env before
          collecting responses.
        </aside>
      )}

      <form
        className={`reasoning-drawer ${selected ? "open" : ""}`}
        onSubmit={submitEvaluation}
      >
        <div className="drawer-grid">
          <label htmlFor="reasoning">
            Briefly, why did you choose this? <span>(Optional)</span>
          </label>
          <textarea
            id="reasoning"
            ref={textareaRef}
            value={reasoning}
            maxLength={4000}
            onChange={(event) => setReasoning(event.target.value)}
            placeholder="Short rationale"
          />
          <div className="drawer-actions">
            <button
              type="submit"
              className="submit-button"
              disabled={!selected || isSubmitting}
            >
              {isSubmitting ? "Submitting" : `Submit ${selected ?? ""} & Next`}
            </button>
          </div>
        </div>
        {error && <p className="error-message">{error}</p>}
      </form>
    </main>
  );
}
