import React, { useState } from "react";
import { useAppData } from "../context/AppDataContext";
import { AuditEvent } from "../types";
import { StatusBadge } from "../components/common/Badge";
import { Button } from "../components/common/Button";
import { Search, Download, Calendar, User, FileText, ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";

export const HistoryPage: React.FC = () => {
  const { auditLogs } = useAppData();
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState("");

  const filteredLogs = auditLogs.filter(
    (log: AuditEvent) =>
      log.imageName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.action.toLowerCase().includes(searchTerm.toLowerCase()) ||
      log.actor.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (log.notes && log.notes.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const exportLogsAsJson = () => {
    const jsonString = JSON.stringify(auditLogs, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `annotation_history_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Header with Back button */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <div>
          <button
            onClick={() => navigate("/dashboard")}
            className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-900 transition-colors mb-1 font-medium"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to Dashboard</span>
          </button>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">Annotation History</h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Chronological audit trail of all uploaded images, AI suggestions, human reviews, edits, and rejections
          </p>
        </div>

        <Button
          variant="secondary"
          size="sm"
          onClick={exportLogsAsJson}
          leftIcon={<Download className="w-4 h-4" />}
        >
          Export History JSON
        </Button>
      </div>

      {/* Filter and Search */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-card flex items-center justify-between gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by image name, notes, status, or annotator…"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:bg-white"
          />
        </div>
        <span className="text-xs text-slate-500 font-mono">
          Total Logged Events: {auditLogs.length}
        </span>
      </div>

      {/* Audit Log Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-700">
            <thead className="bg-slate-50/80 border-b border-slate-200 text-xs font-semibold text-slate-600 uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4">Image</th>
                <th className="py-3 px-4">Action / Event</th>
                <th className="py-3 px-4">Actor</th>
                <th className="py-3 px-4">Source</th>
                <th className="py-3 px-4">Status Transition</th>
                <th className="py-3 px-4">Details / Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-sans text-xs">
              {filteredLogs.map((log: AuditEvent) => (
                <tr key={log.id} className="hover:bg-slate-50/50 transition-colors">
                  {/* Timestamp */}
                  <td className="py-3 px-4 whitespace-nowrap text-slate-600 font-mono text-[11px]">
                    <div className="flex items-center gap-1.5">
                      <Calendar className="w-3.5 h-3.5 text-slate-400" />
                      <span>{new Date(log.timestamp).toLocaleDateString()}</span>
                      <span className="text-slate-400">
                        {new Date(log.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      </span>
                    </div>
                  </td>

                  {/* Image Name */}
                  <td className="py-3 px-4">
                    <button
                      onClick={() => navigate(`/workspace/${log.imageId}`)}
                      className="font-semibold text-slate-900 hover:text-teal-700 underline decoration-slate-300 hover:decoration-teal-600 transition-colors flex items-center gap-1"
                    >
                      <FileText className="w-3.5 h-3.5 text-slate-400" />
                      <span>{log.imageName}</span>
                    </button>
                  </td>

                  {/* Action */}
                  <td className="py-3 px-4 font-medium text-slate-900">{log.action}</td>

                  {/* Actor */}
                  <td className="py-3 px-4 text-slate-600">
                    <span className="flex items-center gap-1">
                      <User className="w-3.5 h-3.5 text-slate-400" />
                      <span>{log.actor}</span>
                    </span>
                  </td>

                  {/* Source */}
                  <td className="py-3 px-4">
                    {log.source ? (
                      <span className="font-mono text-[11px] font-semibold uppercase px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                        {log.source === "ai" ? "AI Suggestion" : log.source === "ai-edited" ? "AI Edited" : "Manual"}
                      </span>
                    ) : (
                      <span className="text-slate-400">—</span>
                    )}
                  </td>

                  {/* Status Transition */}
                  <td className="py-3 px-4">
                    {log.newStatus ? (
                      <StatusBadge status={log.newStatus} size="sm" />
                    ) : (
                      <span className="text-slate-400">—</span>
                    )}
                  </td>

                  {/* Notes */}
                  <td className="py-3 px-4 text-slate-500 max-w-xs truncate" title={log.notes}>
                    {log.notes || "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
