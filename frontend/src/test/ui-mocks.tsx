/**
 * Lightweight mocks for Radix-based shadcn/ui components that use portals
 * or pointer-event APIs unavailable in jsdom.
 *
 * Simple components (Button, Input, Label, Card, Badge) render fine in jsdom
 * and don't need mocking.
 */
import React from "react";

/* ── Select ─────────────────────────────────────────────────────────────── */

export function Select({
  children,
  value,
  onValueChange,
}: {
  children: React.ReactNode;
  value?: string;
  onValueChange?: (v: string) => void;
}) {
  return (
    <div data-testid="mock-select" data-value={value}>
      {React.Children.map(children, (child) =>
        React.isValidElement(child)
          ? React.cloneElement(child as React.ReactElement<Record<string, unknown>>, {
              _value: value,
              _onValueChange: onValueChange,
            })
          : child,
      )}
    </div>
  );
}

export function SelectTrigger({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
  _value?: string;
  _onValueChange?: (v: string) => void;
}) {
  return <button className={className}>{children}</button>;
}

export function SelectContent({ children }: { children: React.ReactNode }) {
  return <div>{children}</div>;
}

export function SelectItem({
  children,
  value,
}: {
  children: React.ReactNode;
  value: string;
}) {
  return <option value={value}>{children}</option>;
}

export function SelectValue() {
  return null;
}

/* ── Tooltip ────────────────────────────────────────────────────────────── */

export function Tooltip({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}

export function TooltipTrigger({
  children,
}: {
  children: React.ReactNode;
  asChild?: boolean;
}) {
  return <>{children}</>;
}

export function TooltipContent({ children }: { children: React.ReactNode }) {
  return <span data-testid="tooltip-content">{children}</span>;
}
